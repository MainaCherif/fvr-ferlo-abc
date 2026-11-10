"""Derived quantities computed from the final ABC-SMC populations: WAIC and deviance profiles."""
import numpy as np
from scipy.special import gammaln
from .model import simulate
from .data import OBS_X, OBS_N, binomial_deviance


def _pointwise_loglik(p):
    """Pointwise binomial log-likelihood of the six counts (no combinatorial constant, as in the deviance)."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return OBS_X * np.log(p) + (OBS_N - OBS_X) * np.log(1 - p)


def particle_predictions(res, Ed, season, D, **sim_kwargs):
    """Seroprevalence (aggregated over ponds) of every particle of a final population: (n_particles, 6)."""
    return np.array([simulate(Ed, season, D, th, **sim_kwargs).mean(axis=0) for th in np.asarray(res["thetas"])])


def waic(res, Ed, season, D, **sim_kwargs):
    """WAIC = -2 (lppd - p_eff); lppd uses the importance weights, p_eff is the weighted sum over the six
    observations of the variance of the pointwise log-likelihood across particles.
    Returns (waic, p_eff)."""
    w = np.asarray(res["weights"], float); w = w / w.sum()
    ll = np.array([_pointwise_loglik(p) for p in particle_predictions(res, Ed, season, D, **sim_kwargs)])  # (n, 6)
    m = ll.max(axis=0)
    lppd = np.sum(m + np.log((w[:, None] * np.exp(ll - m)).sum(axis=0)))
    mean = (w[:, None] * ll).sum(axis=0)
    p_eff = np.sum((w[:, None] * (ll - mean) ** 2).sum(axis=0))
    return float(-2.0 * (lppd - p_eff)), float(p_eff)


def profile(res, Ed, season, D, j, edges, **sim_kwargs):
    """Minimum deviance reached by the final-population particles whose parameter j falls in each bin."""
    th = np.asarray(res["thetas"])
    dev = np.array([binomial_deviance(simulate(Ed, season, D, t, **sim_kwargs).mean(axis=0)) for t in th])
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (th[:, j] >= a) & (th[:, j] < b)
        out.append(float(dev[m].min()) if m.any() else None)
    return out
