"""Observed data and pond geometry."""
import numpy as np
import pandas as pd

# Age-specific IgG seroprevalence, Durand et al. (2020), Table 4: seropositive / tested
OBS_X = np.array([1, 2, 8, 4, 10, 9])
OBS_N = np.array([35, 36, 35, 35, 39, 42])
AGE_MIDPOINTS = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 6.5])


def binomial_deviance(p, x=OBS_X, n=OBS_N):
    """-2 x log-likelihood of the counts (up to a constant combinatorial term)."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-2.0 * np.sum(x * np.log(p) + (n - x) * np.log(1 - p)))


def catalytic_fit():
    """One-parameter catalytic model p(a) = 1 - exp(-lambda a), ML estimate."""
    from scipy.optimize import minimize_scalar
    f = lambda lam: binomial_deviance(1 - np.exp(-lam * AGE_MIDPOINTS))
    r = minimize_scalar(f, bounds=(1e-4, 1.0), method="bounded", options={"xatol": 1e-10})
    return r.x, r.fun


def distance_matrix(coord_csv):
    """Euclidean distances (km) between ponds, local equirectangular projection (Eq. 1)."""
    df = pd.read_csv(coord_csv, encoding="utf-8-sig").sort_values("ID_mare")
    lat, lon = df["Latitude_N"].values, df["Longitude_E"].values
    phi = np.radians(lat.mean())
    x = lon * 111.0 * np.cos(phi)
    y = lat * 111.0
    D = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    return D, df["ID_mare"].tolist()
