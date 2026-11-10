"""Sequential ABC (ABC-SMC) with the binomial deviance as summary statistic.

Algorithm (as described in the manuscript):
  generation 0 : N0 = 350 Latin-hypercube draws from the prior (log scale for beta_v, n_intro,
                 sigma, q), best 30 % kept with uniform weights;
  next         : N = 300 proposals by weighted resampling + Gaussian perturbation
                 (2 x weighted covariance, in log space where relevant); accepted if the deviance
                 is below the median deviance of the previous population; importance weights
                 with a Gaussian kernel;
  stop         : median deviance changes by < 2 % after >= 3 generations, at most 10 generations.
"""
import numpy as np
from scipy.stats import qmc

NAMES = ["beta_direct", "beta_v", "n_intro", "delta_m", "sigma", "alpha", "q"]
BOUNDS = np.array([(0.0, 10.0), (0.005, 0.5), (5.0, 700.0), (1 / 200, 1 / 10),
                   (1e-5, 0.1), (0.05, 3.5), (1e-7, 1e-3)])   # floors sigma>=1e-5, q>=1e-7
LOG = np.array([False, True, True, False, True, False, True])


def run_abc(deviance_fn, active, fixed_values, bounds=BOUNDS, log=LOG, seed=0,
            N0=350, N=300, keep=0.30, max_gen=10, min_gen=3, tol=0.02, verbose=False):
    """deviance_fn(theta[7]) -> float.  active: bool[7] (calibrated parameters);
    fixed_values: values used for the inactive parameters."""
    rng = np.random.default_rng(seed)
    active = np.asarray(active, bool)
    idx = np.where(active)[0]
    d = len(idx)
    with np.errstate(divide="ignore"):
        lo = np.where(log[idx], np.log(bounds[idx, 0]), bounds[idx, 0])
    hi = np.where(log[idx], np.log(bounds[idx, 1]), bounds[idx, 1])

    def to_theta(z):
        th = np.array(fixed_values, float)
        v = np.where(log[idx], np.exp(z), z)
        th[idx] = v
        return th

    sampler = qmc.LatinHypercube(d=d, seed=int(rng.integers(1 << 31)))
    Z = lo + sampler.random(N0) * (hi - lo)
    Dv = np.array([deviance_fn(to_theta(z)) for z in Z])
    all_dev = [Dv.copy()]
    best = (Dv.min(), to_theta(Z[Dv.argmin()]))
    order = np.argsort(Dv)[: int(round(keep * N0))]
    Z, Dv = Z[order], Dv[order]
    W = np.full(len(Z), 1.0 / len(Z))
    history = [dict(gen=0, n=len(Z), median=float(np.median(Dv)), min=float(Dv.min()),
                    median_all=float(np.median(all_dev[0])))]
    for g in range(1, max_gen):
        thr = np.median(Dv)
        mu = np.average(Z, axis=0, weights=W)
        cov = 2.0 * np.atleast_2d(np.cov(Z.T, aweights=W)) + 1e-12 * np.eye(d)
        L = np.linalg.cholesky(cov)
        newZ, newD = [], []
        for _ in range(N):
            while True:
                parent = Z[rng.choice(len(Z), p=W)]
                z = parent + L @ rng.standard_normal(d)
                if np.all(z >= lo) and np.all(z <= hi):
                    break
            dv = deviance_fn(to_theta(z))
            if dv < best[0]:
                best = (dv, to_theta(z))
            if dv < thr:
                newZ.append(z); newD.append(dv)
        if len(newZ) < 10:
            break
        newZ, newD = np.array(newZ), np.array(newD)
        icov = np.linalg.inv(cov)
        dif = newZ[:, None, :] - Z[None, :, :]
        ker = np.exp(-0.5 * np.einsum("ijk,kl,ijl->ij", dif, icov, dif))
        w = 1.0 / (ker @ W)
        w /= w.sum()
        Z, Dv, W = newZ, newD, w
        history.append(dict(gen=g, n=len(Z), median=float(np.median(Dv)), min=float(Dv.min())))
        if verbose:
            print(history[-1])
        if g + 1 >= min_gen and abs(history[-1]["median"] - history[-2]["median"]) / history[-2]["median"] < tol:
            break
    thetas = np.array([to_theta(z) for z in Z])
    return dict(dmin=float(best[0]), theta_best=best[1], thetas=thetas, weights=W, dev=Dv,
                history=history, n_gen=len(history))
