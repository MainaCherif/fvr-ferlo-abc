"""Coupled M-S-E-I-R (hosts, 6 annual age classes) / S-L-I (vectors) model on 14 ponds.

Daily explicit Euler scheme (dt = 1 day, same step as the daily environmental forcing).
Implements Eqs. (M)-(R), (Sv)-(Iv), the force of infection (lambda) with its spatial
decomposition, and the seasonal nomadic introduction (Phi) of the manuscript.
"""
import numpy as np
from numba import njit

NA = 6  # age classes: 0-1, 1-2, 2-3, 3-4, 4-5, >5 years

DEFAULT_FIXED = dict(
    epsilon=1 / 7.0,      # end of latency (1/day)
    gamma=1 / 7.0,        # recovery (1/day)
    life_years=4.0,       # b_h = mu_h = 1/(life_years*365)
    mu_v=1 / 14.0,        # vector mortality
    delta=1 / 10.0,       # extrinsic incubation
    p_early=0.31,         # seasonal split of the introduction (Durand et al. 2020)
    eta_v=5.0,            # vector emergence scale
    S0=20.0,              # initial susceptibles per age class and pond
    Sv0=500.0,            # initial susceptible vectors per pond
)


@njit(cache=True)
def _simulate(Ed, season, D, th, fx, flags, nsub, record):
    dt = 1.0 / nsub
    # th: beta_direct, beta_v, n_intro, delta_m, sigma, alpha, q
    # fx: epsilon, gamma, bh, mu_v, delta, p_early, eta_v, S0, Sv0
    # flags: use_M, use_E, nomad_all_ages
    n = Ed.shape[0]
    T = Ed.shape[1]
    bd, bv, n_intro, dm, sigma, alpha, q = th[0], th[1], th[2], th[3], th[4], th[5], th[6]
    eps, gam, bh, muv, dlt, pe, eta, S0, Sv0 = fx[0], fx[1], fx[2], fx[3], fx[4], fx[5], fx[6], fx[7], fx[8]
    muh = bh
    use_M, use_E, nomad_mode, count_M = flags[0] == 1, flags[1] == 1, flags[2], flags[3] == 1

    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                K[i, j] = np.exp(-alpha * D[i, j])

    M = np.zeros((n, NA)); S = np.full((n, NA), S0); E = np.zeros((n, NA))
    I = np.zeros((n, NA)); R = np.zeros((n, NA))
    Sv = np.full(n, Sv0); Lv = np.zeros(n); Iv = np.zeros(n)
    nm = float(n)
    rec = np.zeros((T // 7 + 1, n)) if record else np.zeros((1, n))

    Ihtot = np.zeros(n); Ntot = np.zeros(n); Rtot = np.zeros(n)
    lam = np.zeros(n)
    for t in range(T):
        s_t = season[t]
        intro = 0.0
        if s_t == 1:
            intro = n_intro / nm * pe
        elif s_t == 2:
            intro = n_intro / nm * (1.0 - pe)
        for sub in range(nsub):
            for i in range(n):
                ni = 0.0; ii = 0.0; ri = 0.0
                for a in range(NA):
                    ni += M[i, a] + S[i, a] + E[i, a] + I[i, a] + R[i, a]
                    ii += I[i, a]; ri += R[i, a]
                Ntot[i] = ni; Ihtot[i] = ii; Rtot[i] = ri
            if record and t % 7 == 0 and sub == 0:
                for i in range(n):
                    s = 0.0
                    for a in range(NA):
                        s += I[i, a] + R[i, a]
                    rec[t // 7, i] = s / Ntot[i]
            for i in range(n):
                nv = Sv[i] + Lv[i] + Iv[i]
                lam_v = bv * Iv[i] / nv if nv > 1e-12 else 0.0
                sp = 0.0
                for j in range(n):
                    sp += K[i, j] * Ihtot[j]
                lam[i] = lam_v + sigma * sp + q + bd * bh * Ihtot[i] / Ntot[i]
            Mn = M.copy(); Sn = S.copy(); En = E.copy(); In = I.copy(); Rn = R.copy()
            for i in range(n):
                Ni = Ntot[i]
                f = Rtot[i] / Ni if use_M else 0.0
                births = bh * Ni
                phi_base = intro * S[i, 2] / Ni  # introduction pressure on the 2-3-year class
                Stot = 0.0
                for a in range(NA):
                    Stot += S[i, a]
                Stot = max(Stot, 1e-12)
                for a in range(NA):
                    if nomad_mode == 0:
                        phi = phi_base if a == 2 else 0.0
                    elif nomad_mode == 1:     # same infection pressure applied to every class
                        phi = intro * S[i, a] / Ni
                    elif nomad_mode == 2:     # baseline total split in proportion to class susceptibles
                        phi = phi_base * S[i, a] / Stot
                    else:                     # baseline total split equally over the 6 classes
                        phi = phi_base / 6.0
                    ao = 1.0 / 365.0 if a < NA - 1 else 0.0
                    dM = -ao * M[i, a] - (dm + muh) * M[i, a]
                    dS = -lam[i] * S[i, a] - ao * S[i, a] + dm * M[i, a] - muh * S[i, a] - phi
                    dE = lam[i] * S[i, a] + phi - eps * E[i, a] - ao * E[i, a] - muh * E[i, a]
                    dI = eps * E[i, a] - gam * I[i, a] - ao * I[i, a] - muh * I[i, a]
                    dR = gam * I[i, a] - ao * R[i, a] - muh * R[i, a]
                    if a > 0:
                        dM += M[i, a - 1] / 365.0; dS += S[i, a - 1] / 365.0
                        dE += E[i, a - 1] / 365.0; dI += I[i, a - 1] / 365.0; dR += R[i, a - 1] / 365.0
                    else:
                        dM += births * f
                        dS += births * (1.0 - f)
                    if not use_E:       # no latency (E stays 0): new infections enter I directly
                        dI += lam[i] * S[i, a] + phi
                    Mn[i, a] = max(M[i, a] + dt * dM, 0.0)
                    Sn[i, a] = max(S[i, a] + dt * dS, 0.0)
                    En[i, a] = max(E[i, a] + dt * dE, 0.0) if use_E else 0.0
                    In[i, a] = max(I[i, a] + dt * dI, 0.0)
                    Rn[i, a] = max(R[i, a] + dt * dR, 0.0)
                ih = Ihtot[i] / Ni
                em = max(0.0, eta * Ed[i, t])
                inf_v = bv * Sv[i] * ih
                Svn = Sv[i] + dt * (em - inf_v - muv * Sv[i])
                Lvn = Lv[i] + dt * (inf_v - dlt * Lv[i] - muv * Lv[i])
                Ivn = Iv[i] + dt * (dlt * Lv[i] - muv * Iv[i])
                Sv[i] = max(Svn, 0.0); Lv[i] = max(Lvn, 0.0); Iv[i] = max(Ivn, 0.0)
            M, S, E, I, R = Mn, Sn, En, In, Rn
    prev = np.zeros((n, NA))
    for i in range(n):
        for a in range(NA):
            tot = M[i, a] + S[i, a] + E[i, a] + I[i, a] + R[i, a]
            num = I[i, a] + R[i, a] + (M[i, a] if count_M else 0.0)
            prev[i, a] = num / tot if tot > 0 else 0.0
    Nfinal = np.zeros(n)
    for i in range(n):
        for a in range(NA):
            Nfinal[i] += M[i, a] + S[i, a] + E[i, a] + I[i, a] + R[i, a]
    return prev, rec, Nfinal


def simulate(Ed, season, D, theta, fixed=None, use_M=True, use_E=True, nomad_all_ages=False, count_M=False,
             nsub=1, record=False):
    """theta = (beta_direct, beta_v, n_intro, delta_m, sigma, alpha, q).
    Returns prevalence per pond and age class (n_ponds x 6) [+ weekly series, final N]."""
    fx = dict(DEFAULT_FIXED)
    if fixed:
        fx.update(fixed)
    bh = 1.0 / (fx["life_years"] * 365.0)
    fxa = np.array([fx["epsilon"], fx["gamma"], bh, fx["mu_v"], fx["delta"], fx["p_early"],
                    fx["eta_v"], fx["S0"], fx["Sv0"]], dtype=np.float64)
    flags = np.array([int(use_M), int(use_E), int(nomad_all_ages), int(count_M)], dtype=np.int64)
    prev, rec, N = _simulate(np.ascontiguousarray(Ed), np.ascontiguousarray(season), np.ascontiguousarray(D),
                             np.asarray(theta, dtype=np.float64), fxa, flags, nsub, record)
    return (prev, rec, N) if record else prev
