import os
import numpy as np
import argparse
from astropy.stats import sigma_clipped_stats


# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run constant shear simulation with MPI"
)
parser.add_argument(
    "--target", type=str, default="g1", help="test target"
)
parser.add_argument(
    "--shear", type=float, default=0.02,
)
parser.add_argument("--layout", type=str, default="grid", help="layout")

args = parser.parse_args()
shear_value = args.shear
test_target = args.target

rng = np.random.default_rng()


pscratch = os.environ.get("PSCRATCH", ".")
outdir = os.path.join(
    pscratch,
    f"constant_shear_{args.layout}",
    test_target,
    f"shear{int(shear_value * 100):02d}",
)

e1s_mode0 = []
e1s_mode1 = []
R1s_mode0 = []
R1s_mode1 = []
NN = []
for rank in range(1024):
    e1s_mode0.append(
        np.load(f'{outdir}/e1s_mode0_rot0_rank{rank:05d}.npy').flatten()
    )
    e1s_mode1.append(
        np.load(f'{outdir}/e1s_mode1_rot0_rank{rank:05d}.npy').flatten()
    )
    R1s_mode0.append(
        np.load(f'{outdir}/R1s_mode0_rot0_rank{rank:05d}.npy').flatten()
    )
    R1s_mode1.append(
        np.load(f'{outdir}/R1s_mode1_rot0_rank{rank:05d}.npy').flatten()
    )
    NN.append(np.load(f'{outdir}/Ns_mode0_rot0_rank{rank:05d}.npy').flatten())

e1s_mode0 = np.hstack(e1s_mode0)
e1s_mode1 = np.hstack(e1s_mode1)
R1s_mode0 = np.hstack(R1s_mode0)
R1s_mode1 = np.hstack(R1s_mode1)
NN = np.hstack(NN)

def m_bootstrap(ep, em, Rp, Rm, Nsample=10000):
    N = len(ep)
    ms = np.zeros(Nsample)
    for i in range(Nsample):
        k = rng.choice(N, N, replace=True)
        new_gamma = np.sum(ep[k] - em[k]) / np.sum(Rp[k] + Rm[k])
        m = new_gamma / shear_value - 1
        ms[i] = m
    return ms

def neff_bootstrap(ep, Rp, Nsample=10000):
    N = len(ep)
    ms = np.zeros(Nsample)
    R = np.average(Rp)
    for i in range(Nsample):
        k = rng.choice(N, N, replace=True)
        ms[i] = np.average(ep[k]) / R
    area = (
        (3900 * 3900) * (0.2 / 60.0) **2.0
    )  # arcmin
    std = np.std(ms)
    neff = (0.26 / std) ** 2.0 / area / N
    return

m = (
    np.sum(e1s_mode1 - e1s_mode0) / np.sum(R1s_mode1 + R1s_mode0)
) / shear_value - 1
c = np.sum(e1s_mode1 + e1s_mode0) / np.sum(R1s_mode1 + R1s_mode0)

area = (
    (3900 * 3900) * (0.2 / 60.0) **2.0
)  # arcmin
mean, median, std = sigma_clipped_stats(
    (e1s_mode1 / NN) / (np.sum(R1s_mode1) / np.sum(NN)),
    sigma=5.0
)
neff_bootstrap(e1s_mode1, R1s_mode1)
neff = (0.26 / std) ** 2.0 / area
print(neff)
print(np.average(NN) / area)
print(np.sum(NN))

ms = m_bootstrap(e1s_mode1, e1s_mode0, R1s_mode1, R1s_mode0)
ord_ms = np.sort(ms)
sigma_m = (ord_ms[8413] - ord_ms[1587]) / 2.0
mean_m = np.mean(ms)
print(m, mean_m, sigma_m)
print(c)
