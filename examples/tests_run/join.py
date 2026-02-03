#!/usr/bin/env python3
import argparse
import gc
import os
from typing import List, Optional

import astropy.table as astTable
import pyarrow as pa
import pyarrow.parquet as pq
from mpi4py import MPI  # type: ignore

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

# ------------------------------
# Argument Parsing
# ------------------------------
parser = argparse.ArgumentParser(
    description="Convert FITS catalogs to Parquets partitioned by group_id",
    allow_abbrev=False,
)
parser.add_argument("--target", type=str, default="g1", help="test target")
parser.add_argument(
    "--mode", type=int, default=0, choices=[40, 0, 27, 9, 3, 1, 36, 4, 80],
    help="40:++++;0:----;27:+---;9:-+--;3:--+-;1:---+;36:++--;4:--++;80:0000"
)
parser.add_argument(
    "--rot", type=int, default=0, choices=[0, 1], help="rotation id",
)
parser.add_argument(
    "--start", type=int, default=0, help="start group_id (inclusive)",
)
parser.add_argument(
    "--end", type=int, default=2, help="end group_id (exclusive)",
)
parser.add_argument("--shear", type=float, default=0.02, help="Shear value")
parser.add_argument("--kappa", type=float, default=0.00, help="Kappa value")
parser.add_argument(
    "--layout",
    type=str, default="grid",
    choices=["grid", "random"],
    help="layout",
)
parser.add_argument(
    "--skip-existing", action="store_true",
    help="Skip seeds whose Parquet file already exists."
)
args, unknown = parser.parse_known_args()
if unknown:
    print("[warn] Ignoring unknown args:", unknown)

shear_mode = int(args.mode)
shear_value = args.shear
kappa_value = args.kappa
rot_id = args.rot
test_target = args.target
group_start = args.start
group_end = args.end

# ------------------------------
# MPI (or single-process) info
# ------------------------------
if RANK == 0:
    if SIZE == 1:
        print("[Info] Running single-process (no mpirun/srun needed).")
    else:
        print(f"[Info] Running with MPI across {SIZE} ranks.")
if group_end - group_start <= 0:
    raise ValueError(
        f"Invalid group range: start={group_start}, end={group_end}"
    )

# ------------------------------
# Paths
# ------------------------------
pscratch = os.environ.get("PSCRATCH", ".")
fits_root = os.path.join(
    pscratch,
    f"constant_shear_{args.layout}",
    test_target,
    f"shear{int(shear_value * 100):02d}",
    f"mode{shear_mode}",
)
if not os.path.isdir(fits_root):
    raise FileNotFoundError(f"Input FITS dir not found: {fits_root}")

pq_root = os.path.join(
    pscratch,
    "parquet",
    f"constant_shear_{args.layout}",
    test_target,
    f"shear{int(shear_value * 100):02d}",
    f"mode{shear_mode}",
)
os.makedirs(pq_root, exist_ok=True)

truth_root = os.path.join(
    pscratch,
    "parquet",
    f"constant_shear_{args.layout}",
    test_target,
    f"shear{int(shear_value * 100):02d}",
    f"truth-mode{shear_mode}",
)
os.makedirs(truth_root, exist_ok=True)


# ------------------------------
# Helpers
# ------------------------------
def _read_and_stack(sim_seed: int) -> astTable.Table:
    """Read main + per-band FITS for a sim_seed; hstack columns (exact)."""
    detfname = os.path.join(fits_root, f"cat-{sim_seed:05d}.fits")
    if not os.path.exists(detfname):
        return None
    detection = astTable.Table.read(detfname)

    data_all: List[astTable.Table] = [detection]
    for band in "ugrizy":
        fname = os.path.join(fits_root, f"cat-{sim_seed:05d}-{band}.fits")
        if os.path.exists(fname):
            data_all.append(astTable.Table.read(fname))

    # If you need strict presence of all bands, replace with join_type="exact"
    dd = astTable.hstack(data_all, join_type="exact")
    dd["sim_seed"] = sim_seed
    return dd


def _read_truth(sim_seed: int) -> Optional[astTable.Table]:
    """Read the truth FITS for a sim_seed (if present)."""
    truthfname = os.path.join(fits_root, f"truth-{sim_seed:05d}.fits")
    if not os.path.exists(truthfname):
        return None

    truth = astTable.Table.read(truthfname)
    truth = truth.copy()

    if "truth_index" not in truth.colnames and "indices" in truth.colnames:
        truth.rename_column("indices", "truth_index")

    truth["sim_seed"] = sim_seed
    return truth


def _astropy_to_arrow(tab: astTable.Table) -> pa.Table:
    """Astropy Table -> Arrow Table (native endianness, no index)."""
    df = tab.to_pandas(index=None)
    # Ensure native-endian for numeric columns
    for c in df.columns:
        a = df[c].to_numpy()
        if getattr(a.dtype, "byteorder", "=") not in ("=", "|"):
            df[c] = a.byteswap().newbyteorder()
    return pa.Table.from_pandas(df, preserve_index=False)


def _group_partition_dir(base_dir: str, group_id: int) -> str:
    """Return the hive-style partition directory for a group_id."""
    return os.path.join(base_dir, f"group_id={group_id}")


def _write_one_group_parquet(
    base_dir: str, t: pa.Table, group_id: int, overwrite: bool
):
    """Write one Parquet file per group_id to .../group_id=.../data.parquet."""
    dir_path = _group_partition_dir(base_dir, group_id)
    os.makedirs(dir_path, exist_ok=True)

    out_path = os.path.join(dir_path, "data.parquet")
    if (not overwrite) and os.path.exists(out_path):
        return

    tmp_path = os.path.join(dir_path, ".data.parquet.tmp")
    pq.write_table(
        t,
        tmp_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    # Atomic rename to avoid partial files on crashes
    os.replace(tmp_path, out_path)


# ------------------------------
# Work loop (unique seeds per rank if MPI)
# ------------------------------
for group_id in range(group_start + RANK, group_end, SIZE):
    group_dir = _group_partition_dir(pq_root, group_id)
    group_path = os.path.join(group_dir, "data.parquet")
    truth_group_dir = _group_partition_dir(truth_root, group_id)
    truth_group_path = os.path.join(truth_group_dir, "data.parquet")

    if (
        args.skip_existing
        and os.path.exists(group_path)
        and os.path.exists(truth_group_path)
    ):
        print("Already has output files")
        continue

    tables: List[pa.Table] = []
    truth_tables: List[pa.Table] = []

    seed_start = group_id * 100
    seed_end = (group_id + 1) * 100

    for sim_seed in range(seed_start, seed_end):
        tab = _read_and_stack(sim_seed)
        if tab is not None:
            tables.append(_astropy_to_arrow(tab))
            del tab

        truth_tab = _read_truth(sim_seed)
        if truth_tab is not None:
            truth_tables.append(_astropy_to_arrow(truth_tab))
            del truth_tab
        gc.collect()

    if tables:
        combined = pa.concat_tables(tables, promote=True).combine_chunks()
        _write_one_group_parquet(
            pq_root, combined, group_id, overwrite=not args.skip_existing
        )
        del combined
        del tables
    else:
        print("Cannot find any input files")

    if truth_tables:
        truth_combined = pa.concat_tables(
            truth_tables, promote=True,
        ).combine_chunks()
        _write_one_group_parquet(
            truth_root, truth_combined, group_id,
            overwrite=not args.skip_existing
        )
        del truth_combined
        del truth_tables
    else:
        print("Cannot find any input files")

    gc.collect()

# Ensure all ranks finish (no-op in single process)
COMM.Barrier()
if RANK == 0:
    print(f"Done. Parquet dataset at: {pq_root}")
