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
    description="Convert FITS catalogs to a Parquet partitioned by sim_seed."
)
parser.add_argument("--target", type=str, default="g1", help="test target")
parser.add_argument(
    "--mode", type=int, default=0, choices=[40, 0, 27, 9, 3, 1, 36, 4, 80],
    help="40:++++;0:----;27:+---;9:-+--;3:--+-;1:---+;36:++--;4:--++;80:0000"
)
parser.add_argument(
    "--rot", type=int, default=0, choices=[0, 1], help="rotation id",
)
parser.add_argument("--start", type=int, default=0, help="start id (inclusive)")
parser.add_argument("--end", type=int, default=2, help="end id (exclusive)")
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
istart = args.start
iend = args.end

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
if iend - istart <= 0:
    raise ValueError(f"Invalid range: start={istart}, end={iend}")

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


def _seed_partition_dir(base_dir: str, sim_seed: int) -> str:
    """Return the hive-style partition directory for a sim_seed."""
    bucket = int(sim_seed // 100)
    return os.path.join(
        base_dir,
        f"sim_seed_bucket={bucket}", f"sim_seed={sim_seed}",
    )


def _write_one_seed_parquet(
    base_dir: str, t: pa.Table, sim_seed: int, overwrite: bool
):
    """Write a single seed to .../sim_seed_bucket=.../sim_seed=.../data.parquet
    atomically."""
    dir_path = _seed_partition_dir(base_dir, sim_seed)
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
for i in range(istart, iend):
    sim_seed = i * size + rank  # same distribution you used before
    try:
        tab = _read_and_stack(sim_seed)
    except FileNotFoundError:
        continue

    t = _astropy_to_arrow(tab)

    _write_one_seed_parquet(
        pq_root, t, sim_seed, overwrite=not args.skip_existing
    )

    # free memory early
    del tab, t
    gc.collect()

# Ensure all ranks finish (no-op in single process)
_barrier()
if rank == 0:
    print(f"Done. Parquet dataset at: {pq_root}")
