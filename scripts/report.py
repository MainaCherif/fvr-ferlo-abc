"""Print every number reported in the Results of the manuscript, from results/results.json.

usage: python scripts/report.py [results/results.json] --ndvi <NDVI csv> --coords <ponds csv>
The manuscript tables and text are written from this output (nothing is typed by hand).
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fvr.data import catalytic_fit, distance_matrix
from fvr.forcing import build_forcing
from fvr.model import simulate
from fvr.analysis import waic, profile
from fvr import abc_smc

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="?", default="results/results.json")
ap.add_argument("--ndvi", required=True); ap.add_argument("--coords", required=True)
A = ap.parse_args()
R = json.load(open(A.results))
Ed, season, _ = build_forcing(A.ndvi, "NDVI")
D, _ = distance_matrix(A.coords)
G7 = dict(gamma=1 / 7.0)
d = lambda k: R[k]["dmin"]
two = lambda a, b: f"{d(a):.2f} / {d(b):.2f}"
lam, dcat = catalytic_fit()
STRUCT = {"MSEIR": {}, "SIR": dict(use_M=False, use_E=False), "MSIR": dict(use_E=False), "SEIR": dict(use_M=False)}
K = {"MSEIR": 7, "SIR": 6, "MSIR": 7, "SEIR": 6}


def wq(x, w, q):
    o = np.argsort(x); x, w = np.asarray(x)[o], np.asarray(w)[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum(); return float(np.interp(q, c, x))


print(f"Catalytic: lambda={lam:.3f}, D={dcat:.2f}, AIC={dcat + 2:.1f}")
print("\nTable 1 (Dmin gamma=1/7 two seeds | gamma=1/5 two seeds | AIC | WAIC mean of two seeds)")
rows = [("MSEIR no space", "spatial/MSEIR/none", 4, None), ("SIR full", "struct/SIR/full", 6, "SIR"), ("SEIR full", "struct/SEIR/full", 6, "SEIR"),
        ("MSEIR distance only", "spatial/MSEIR/distance", 6, None), ("MSEIR full", "struct/MSEIR/full", 7, "MSEIR"), ("MSIR full", "struct/MSIR/full", 7, "MSIR")]
for lab, pre, k, st in rows:
    aic = min(d(f"{pre}/g7/s1"), d(f"{pre}/g7/s2")) + 2 * k
    w = ""
    if st:
        ws = [waic(R[f"{pre}/g7/s{s}"], Ed, season, D, fixed=G7, **STRUCT[st]) for s in (1, 2)]
        w = f"WAIC {np.mean([x[0] for x in ws]):.1f} (each seed {ws[0][0]:.2f}, {ws[1][0]:.2f}; p_eff {ws[0][1]:.2f}, {ws[1][1]:.2f})"
    print(f"  {lab:22s} k={k}  {two(pre + '/g7/s1', pre + '/g7/s2')} | {two(pre + '/g5/s1', pre + '/g5/s2')} | AIC {aic:.1f} {w}")
g7 = [d(f"struct/{s}/full/g7/s{i}") for s in STRUCT for i in (1, 2)]
print(f"  span of the full variants, gamma=1/7: {min(g7):.2f} - {max(g7):.2f} ({max(g7) - min(g7):.2f})")

print("\nTable 2 (spatial variants, gamma=1/7)")
print("  none", two("spatial/MSEIR/none/g7/s1", "spatial/MSEIR/none/g7/s2"), "| background", two("spatial/MSEIR/background/g7/s1", "spatial/MSEIR/background/g7/s2"),
      "| distance", two("spatial/MSEIR/distance/g7/s1", "spatial/MSEIR/distance/g7/s2"), "| full", two("struct/MSEIR/full/g7/s1", "struct/MSEIR/full/g7/s2"))
pl = [d(f"placebo/ndvi/perm{i}") for i in range(5)]; pf = [d(f"placebo/ndvi_full/perm{i}") for i in range(5)]
print(f"  distance only, 5 permutations {min(pl):.2f} - {max(pl):.2f}; full, 5 permutations {min(pf):.2f} - {max(pf):.2f}")
print("  NDWI: distance", two("ndwi/MSEIR/distance/s1", "ndwi/MSEIR/distance/s2"), "| none", two("ndwi/MSEIR/none/s1", "ndwi/MSEIR/none/s2"),
      "| 3 permutations", " / ".join(f"{d(f'ndwi/placebo/perm{i}'):.2f}" for i in range(3)),
      f"| SIR {d('ndwi/SIR/full'):.2f} | full {d('ndwi/MSEIR/full'):.2f}")

print("\nAblation checks")
print("  latency 1.5 d: MSEIR / SEIR / SIR / MSEIR no space:", " / ".join(f"{d(k):.2f}" for k in ("latency1.5/MSEIR/full", "latency1.5/SEIR/full", "latency1.5/SIR/full", "latency1.5/MSEIR/none")))
print(f"  M counted as seropositive: {d('countM/MSEIR/full'):.2f}")
for s in (1, 2):
    r = R[f"struct/MSEIR/full/g7/s{s}"]
    print(f"  seed {s}: median 1/delta_m = {1 / wq(np.array(r['thetas'])[:, 3], r['weights'], .5):.1f} d")

print("\nTable 4 (sensitivity: MSEIR full / SIR full / MSEIR none)")
for s in ("ref", "A", "B", "C", "D", "literature"):
    print(f"  {s:10s}", " / ".join(f"{d(f'sens/{s}/{m}'):.2f}" for m in ("MSEIR_full", "SIR_full", "MSEIR_none")))
v = [d(f"sens/{s}/{m}") for s in ("A", "B", "C", "D", "literature") for m in ("MSEIR_full", "SIR_full", "MSEIR_none")]
print(f"  range over the 15 combinations: {min(v):.2f} - {max(v):.2f}")

print("\nNomadic introduction")
for mode in (1, 2, 3):
    print(f"  all ages, definition {mode}: bound>=5: {two(f'nomad/allages_def{mode}/s1', f'nomad/allages_def{mode}/s2')} | bound>=0.01: {two(f'nomad/allages_def{mode}_lown/s1', f'nomad/allages_def{mode}_lown/s2')}")
print("  2-3-year class, bound>=0.01:", two("nomad/concentrated_lown/s1", "nomad/concentrated_lown/s2"), "| bound>=5 (reference):", two("struct/MSEIR/full/g7/s1", "struct/MSEIR/full/g7/s2"))
print("  p_early 0.02 / 0.6 / 0.88:", " / ".join(f"{d(f'nomad/pearly{p}'):.2f}" for p in (0.02, 0.6, 0.88)))

print("\nSimulated seroprevalence by pond at the end (best particle, %)")
for nm, key in (("full", "struct/MSEIR/full/g7/s1"), ("no coupling", "spatial/MSEIR/none/g7/s1")):
    _, rec, _ = simulate(Ed, season, D, R[key]["theta_best"], fixed=G7, record=True)
    f = rec[-1] * 100
    print(f"  {nm}: mean {f.mean():.1f}, sd {f.std(ddof=1):.1f}, min {f.min():.1f}, max {f.max():.1f}")

names = abc_smc.NAMES
print("\nIdentifiability")
for key in ("ident/orig", "ident/lown", "ident/wide_literature"):
    r = R[key]; th = np.array(r["thetas"]); w = np.array(r["weights"])
    print(f"{key}: Dmin={r['dmin']:.2f}, final particles={len(w)}, generations={r['n_gen']}")
    for j in (1, 2, 3, 4, 5, 6):
        print(f"  {names[j]:8s} median={wq(th[:, j], w, .5):.3g}  95% [{wq(th[:, j], w, .025):.3g}; {wq(th[:, j], w, .975):.3g}]  max={th[:, j].max():.3g}")
print("Profiles (minimum deviance of the final-population particles per bin), key ident/orig")
r = R["ident/orig"]
fmt = lambda v: [None if x is None else round(x, 2) for x in v]
print("  sigma  bins [1e-5,1e-4,3e-4,1e-3,1e-1):", fmt(profile(r, Ed, season, D, 4, [1e-5, 1e-4, 3e-4, 1e-3, 1e-1], fixed=G7)))
print("  alpha  bins [0.05,1,2,3,3.5]:           ", fmt(profile(r, Ed, season, D, 5, [0.05, 1, 2, 3, 3.5001], fixed=G7)))
print("  n_intro bins [5,6,8,10,12,700):         ", fmt(profile(r, Ed, season, D, 2, [5, 6, 8, 10, 12, 700], fixed=G7)))
print("  n_intro bins, bound 0.01 [0.01,1,2,5,10,12,700):", fmt(profile(R["ident/lown"], Ed, season, D, 2, [0.01, 1, 2, 5, 10, 12, 700], fixed=G7)))
