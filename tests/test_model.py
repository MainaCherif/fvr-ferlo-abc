"""Sanity checks. Run with:  python -m pytest tests  (needs data_restricted/ for the geometry test)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fvr.data import catalytic_fit, distance_matrix
from fvr.forcing import build_forcing
from fvr.model import simulate

ROOT = os.path.join(os.path.dirname(__file__), "..", "data_restricted")
THETA = [4.8, 0.036, 7.2, 0.045, 3e-5, 1.9, 4e-6]


def test_catalytic_reference():
    lam, dev = catalytic_fit()
    assert abs(lam - 0.052) < 1e-3 and abs(dev - 180.78) < 0.01


def test_geometry_and_forcing():
    D, _ = distance_matrix(os.path.join(ROOT, "ponds_coordinates.csv"))
    assert abs(D.max() - 9.08) < 0.01 and abs(D[D > 0].min() - 0.03) < 0.01
    Ed, season, ids = build_forcing(os.path.join(ROOT, "appeears_ndvi.csv"))
    assert Ed.shape == (14, 2922) and 0 <= Ed.min() and Ed.max() <= 1


def test_conservation_and_step_convergence():
    D, _ = distance_matrix(os.path.join(ROOT, "ponds_coordinates.csv"))
    Ed, season, _ = build_forcing(os.path.join(ROOT, "appeears_ndvi.csv"))
    p1, _, N = simulate(Ed, season, D, THETA, record=True)
    assert np.allclose(N, 120.0, atol=1e-6)                 # b_h = mu_h: population constant
    p4 = simulate(Ed, season, D, THETA, nsub=4)             # smaller time step
    assert np.abs(p1 - p4).max() < 0.01
