#!/usr/bin/env python3
import argparse
import gc
import os
from typing import List

import astropy.table as astTable
import pyarrow as pa
import pyarrow.parquet as pq


# --- Optional MPI: works without mpirun/srun or even mpi4py installed ---
try:
    from mpi4py import MPI  # type: ignore
    _COMM = MPI.COMM_WORLD
    _RANK = _COMM.Get_rank()
    _SIZE = _COMM.Get_size()

    def _barrier():
        _COMM.Barrier()
except Exception:
    class _FakeComm:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1
    _COMM = _FakeComm()
    _RANK = 0
    _SIZE = 1

    def _barrier():
        return


# ------------------------------
# Argument Parsing
# ------------------------------
parser = argparse.ArgumentParser(
    description="Convert FITS catalogs to Parquet files partitioned by group_id (100 seeds).",
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
comm = _COMM
rank = _RANK
size = _SIZE
if rank == 0:
    if size == 1:
        print("[Info] Running single-process (no mpirun/srun needed).")
    else:
        print(f"[Info] Running with MPI across {size} ranks.")
if group_end - group_start <= 0:
    raise ValueError(f"Invalid group range: start={group_start}, end={group_end}")

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


# ------------------------------
# Helpers
# ------------------------------
def _read_and_stack(sim_seed: int) -> astTable.Table:
    """Read main + per-band FITS for a sim_seed; hstack columns (exact)."""
    detfname = os.path.join(fits_root, f"cat-{sim_seed:05d}.fits")
    if not os.path.exists(detfname):
        raise FileNotFoundError(detfname)
    detection = astTable.Table.read(detfname)

    data_all: List[astTable.Table] = [detection]
    for band in "grizy":
        fname = os.path.join(fits_root, f"cat-{sim_seed:05d}-{band}.fits")
        if not os.path.exists(fname):
            # If some bands are optional, just skip them. Otherwise, raise.
            continue
        data_all.append(astTable.Table.read(fname))

    # If you need strict presence of all bands, replace with join_type="exact"
    return astTable.hstack(data_all, join_type="exact")


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
    """Write one Parquet file per group_id to .../group_id=.../data.parquet atomically."""
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
for group_id in range(group_start + rank, group_end, size):
    group_dir = _group_partition_dir(pq_root, group_id)
    group_path = os.path.join(group_dir, "data.parquet")

    if args.skip_existing and os.path.exists(group_path):
        continue

    tables: List[pa.Table] = []

    seed_start = group_id * 100
    seed_end = (group_id + 1) * 100

    for sim_seed in range(seed_start, seed_end):
        try:
            tab = _read_and_stack(sim_seed)
        except FileNotFoundError:
            continue

        t = _astropy_to_arrow(tab)
        tables.append(t)

        # free memory early for astropy table
        del tab
        gc.collect()

    if not tables:
        continue

    combined = pa.concat_tables(tables, promote=True).combine_chunks()

    _write_one_group_parquet(
        pq_root, combined, group_id, overwrite=not args.skip_existing
    )

    del tables, combined
    gc.collect()

# Ensure all ranks finish (no-op in single process)
_barrier()
if rank == 0:
    print(f"Done. Parquet dataset at: {pq_root}")
