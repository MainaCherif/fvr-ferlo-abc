"""Experiment definitions (model variants) built on top of model.py and abc_smc.py."""
import numpy as np
from .model import simulate
from .data import binomial_deviance
from . import abc_smc

THETA_DEFAULT_FIXED = np.array([4.8, 0.036, 7.2, 1 / 40, 0.0, 1.0, 0.0])


def make_variant(Ed, season, D, *, use_M=True, use_E=True, spatial="full", fixed=None,
                 nomad_all_ages=False, count_M=False, beta_direct_fixed=None):
    """spatial in {'none','background','distance','full'}.
    Returns (deviance_fn, active mask, fixed theta)."""
    active = np.array([True, True, True, use_M, False, False, False])
    if spatial in ("distance", "full"):
        active[4] = True; active[5] = True
    if spatial in ("background", "full"):
        active[6] = True
    if beta_direct_fixed is not None:
        active[0] = False
    fx = THETA_DEFAULT_FIXED.copy()
    if beta_direct_fixed is not None:
        fx[0] = beta_direct_fixed

    def dev(theta):
        p = simulate(Ed, season, D, theta, fixed=fixed, use_M=use_M, use_E=use_E,
                     nomad_all_ages=nomad_all_ages, count_M=count_M)
        return binomial_deviance(p.mean(axis=0))
    return dev, active, fx


def calibrate(Ed, season, D, seed=0, verbose=False, bounds=None, abc_kwargs=None, **variant):
    dev, active, fx = make_variant(Ed, season, D, **variant)
    kw = dict(abc_kwargs or {})
    res = abc_smc.run_abc(dev, active, fx, seed=seed, verbose=verbose,
                          **({"bounds": bounds} if bounds is not None else {}), **kw)
    res["k"] = int(active.sum())
    return res


def permute_distances(D, rng):
    """Randomly permute the pairwise distances between ponds (placebo test)."""
    n = D.shape[0]
    iu = np.triu_indices(n, 1)
    vals = rng.permutation(D[iu])
    P = np.zeros_like(D)
    P[iu] = vals
    return P + P.T
