import os
import numpy as np
import argparse


# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run constant shear simulation with MPI"
)
parser.add_argument("--root", type=str, default="./", help="root dir")
parser.add_argument(
    "--mode", type=int, default=0, choices=[0, 1, 2],
    help="Shear mode: 0: g=-shear, 1: g=shear, 2: g=0.00"
)
parser.add_argument(
    "--shear", type=float, default=0.02,
)
args = parser.parse_args()

shear_value = args.shear
root_dir = args.root

rng = np.random.default_rng()

outdir = os.path.join(
    os.environ["SCRATCH"],
    "isolated",
    root_dir,
    f"anacal_blends_shear{int(shear_value * 100):02d}"
)

e1s_mode0 = np.load(f'{outdir}/e2s_mode0_rot0.npy').flatten()
e1s_mode1 = np.load(f'{outdir}/e2s_mode1_rot0.npy').flatten()

R1s_mode0 = np.load(f'{outdir}/R2s_mode0_rot0.npy').flatten()
R1s_mode1 = np.load(f'{outdir}/R2s_mode1_rot0.npy').flatten()
print(e1s_mode0.shape)

def m_bootstrap(ep, em, Rp, Rm, Nsample=10000):
    N = len(ep)
    ms = np.zeros(Nsample)
    for i in range(Nsample):
        k = rng.choice(N, N, replace=True)
        new_gamma = np.sum(ep[k] - em[k]) / np.sum(Rp[k] + Rm[k])
        m = new_gamma / shear_value - 1
        ms[i] = m
    return ms

m = (
    np.sum(e1s_mode1 - e1s_mode0) / np.sum(R1s_mode1 + R1s_mode0)
) / shear_value - 1

ms = m_bootstrap(e1s_mode1, e1s_mode0, R1s_mode1, R1s_mode0)
ord_ms = np.sort(ms)
sigma_m = (ord_ms[9750] - ord_ms[250]) / 4
mean_m = np.mean(ms)
print(m, mean_m, sigma_m)
