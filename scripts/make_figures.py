"""Draw the result figures of the manuscript from results/results.json.

usage: python scripts/make_figures.py --ndvi <NDVI csv> --coords <ponds csv> [--results results/results.json] [--out figures]
Figures: fig11 (fit and deviances), fig13 (age profile), fig14 (dynamics), fig15 (map), fig16 (placebo),
fig17 (identifiability), fig18 (ABC-SMC convergence).
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fvr.data import OBS_X, OBS_N, AGE_MIDPOINTS, catalytic_fit, distance_matrix
from fvr.forcing import build_forcing
from fvr.model import simulate
from fvr import abc_smc

ap = argparse.ArgumentParser()
ap.add_argument("--ndvi", required=True); ap.add_argument("--coords", required=True)
ap.add_argument("--results", default="results/results.json"); ap.add_argument("--out", default="figures")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
R = json.load(open(A.results))
D, ids = distance_matrix(A.coords)
Ed, season, _ = build_forcing(A.ndvi, "NDVI")
G7 = dict(gamma=1 / 7.0)
AGES = ["0-1", "1-2", "2-3", "3-4", "4-5", ">5"]
BLUE, RED, GREY, GREEN = "#1f5fa8", "#c0392b", "#7f8c8d", "#2e8b57"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})


def wilson(x, n, z=1.96):
    p = x / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def best_of(*keys):
    return min((R[k] for k in keys), key=lambda r: r["dmin"])


def sim(theta, **kw):
    return simulate(Ed, season, D, theta, fixed=G7, **kw)


def save(fig, name):
    fig.savefig(os.path.join(A.out, name), dpi=200, bbox_inches="tight"); plt.close(fig)


obs = OBS_X / OBS_N
lo, hi = wilson(OBS_X, OBS_N)
lam, dcat = catalytic_fit()
full = best_of("struct/MSEIR/full/g7/s1", "struct/MSEIR/full/g7/s2")
sirf = best_of("struct/SIR/full/g7/s1", "struct/SIR/full/g7/s2")

# ---- fig11: fit and deviances
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1, 1.25]))
a = ax[0]
a.errorbar(AGE_MIDPOINTS, obs, yerr=[obs - lo, hi - obs], fmt="o", color="k", capsize=3, label="Observed (Wilson 95% CI)")
t = np.linspace(0, 7, 100)
a.plot(t, 1 - np.exp(-lam * t), color=GREY, label=f"Catalytic (D = {dcat:.1f})")
a.plot(AGE_MIDPOINTS, sim(full["theta_best"]).mean(0), "-s", color=BLUE, ms=4, label=f"Full M-S-E-I-R (D = {full['dmin']:.1f})")
a.plot(AGE_MIDPOINTS, sim(sirf["theta_best"], use_M=False, use_E=False).mean(0), "-^", color=RED, ms=4, label=f"Full S-I-R (D = {sirf['dmin']:.1f})")
a.set_xlabel("Age (years)"); a.set_ylabel("IgG seroprevalence"); a.set_title("(a)", loc="left"); a.legend(fontsize=8, frameon=False)
b = ax[1]
groups = [("MSEIR", "M-S-E-I-R"), ("SIR", "S-I-R"), ("MSIR", "M-S-I-R"), ("SEIR", "S-E-I-R")]
labels, x = [], 0
for g in ("g7", "g5"):
    for sname, lab in groups:
        for s in ("s1", "s2"):
            b.plot(x, R[f"struct/{sname}/full/{g}/{s}"]["dmin"], "o", color=BLUE, ms=4)
        labels.append((x, f"{lab}\n{'1/7' if g == 'g7' else '1/5'}")); x += 1
    x += 0.5
for sp, lab in (("none", "no coupling"), ("background", "background"), ("distance", "distance")):
    for s in ("s1", "s2"):
        b.plot(x, R[f"spatial/MSEIR/{sp}/g7/{s}"]["dmin"], "o", color=GREEN, ms=4)
    labels.append((x, lab)); x += 1
for i in range(5):
    b.plot(x, R[f"placebo/ndvi/perm{i}"]["dmin"], "o", color=RED, ms=4)
labels.append((x, "5 permutations")); x += 1
b.axhline(dcat, ls="--", color="k", lw=1); b.text(0, dcat + 0.02, "catalytic", fontsize=8)
b.set_xticks([p for p, _ in labels]); b.set_xticklabels([l for _, l in labels], rotation=60, ha="right", fontsize=7)
b.set_ylabel("Minimum binomial deviance"); b.set_title("(b)", loc="left")
fig.tight_layout(); save(fig, "fig11_ajustement_comparaison_en.png")

# ---- fig13: age profile with the best particle, range across ponds
p = sim(full["theta_best"])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
a = ax[0]
a.errorbar(np.arange(6), obs, yerr=[obs - lo, hi - obs], fmt="o", color="k", capsize=3, label="Observed (Wilson 95% CI)")
a.fill_between(np.arange(6), p.min(0), p.max(0), color=BLUE, alpha=0.3)
a.plot(np.arange(6), p.mean(0), "-s", color=BLUE, ms=4, label="Full M-S-E-I-R, mean across ponds")
a.set_xticks(np.arange(6)); a.set_xticklabels(AGES); a.set_xlabel("Age class (years)"); a.set_ylabel("IgG seroprevalence")
a.set_title("(a)", loc="left"); a.legend(fontsize=8, frameon=False)
b = ax[1]
b.bar(np.arange(6), OBS_N, color=GREY)
for i in range(6):
    b.text(i, OBS_N[i] + 0.5, f"{OBS_X[i]}/{OBS_N[i]}", ha="center", fontsize=9)
b.set_xticks(np.arange(6)); b.set_xticklabels(AGES); b.set_xlabel("Age class (years)"); b.set_ylabel("Number tested")
b.set_ylim(0, 50); b.set_title("(b)", loc="left")
fig.tight_layout(); save(fig, "fig13_serologie_age_en.png")

# ---- fig14: simulated dynamics
_, rec, _ = simulate(Ed, season, D, full["theta_best"], fixed=G7, record=True)
days = np.arange(rec.shape[0]) * 7 / 365.0 + 2008
fig, a = plt.subplots(figsize=(9, 3.8))
for i in range(rec.shape[1]):
    a.plot(days, rec[:, i], color=BLUE, alpha=0.25, lw=0.8)
a.plot(days, rec.mean(1), color=BLUE, lw=2, label="Simulated seroprevalence (mean of ponds)")
a.set_xlabel("Year"); a.set_ylabel("Seroprevalence (all ages)")
b = a.twinx(); b.spines["right"].set_visible(True)
tt = 2008 + np.arange(Ed.shape[1]) / 365.0
b.plot(tt, Ed.mean(0), color=GREY, lw=1, label="Mean forcing $E_d$"); b.set_ylabel("Environmental forcing")
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.93), fontsize=8, frameon=False)
fig.tight_layout(); save(fig, "fig14_dynamique_en.png")

# ---- fig15: map of final simulated seroprevalence
co = pd.read_csv(A.coords, encoding="utf-8-sig").sort_values("ID_mare")
fin = rec[-1]
fig, a = plt.subplots(figsize=(5.8, 5))
sc = a.scatter(co["Longitude_E"], co["Latitude_N"], c=fin * 100, s=260, cmap="viridis", edgecolor="k")
for i, (x_, y_) in enumerate(zip(co["Longitude_E"], co["Latitude_N"])):
    a.text(x_, y_, str(i + 1), ha="center", va="center", color="w", fontsize=8)
plt.colorbar(sc, label="Simulated seroprevalence at the end (%)")
a.set_xlabel("Longitude E"); a.set_ylabel("Latitude N")
fig.tight_layout(); save(fig, "fig15_carte_mares_en.png")

# ---- fig16: placebo test
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for a, idx, perms, real, none, title in (
        (ax[0], "ndvi", [f"placebo/ndvi/perm{i}" for i in range(5)], "spatial/MSEIR/distance/g7/s1", "spatial/MSEIR/none/g7/s1", "NDVI"),
        (ax[1], "ndwi", [f"ndwi/placebo/perm{i}" for i in range(3)], "ndwi/MSEIR/distance/s1", "ndwi/MSEIR/none/s1", "NDWI")):
    a.plot([0] * 2, [R[real]["dmin"]] * 2, "o", color=BLUE, label="Real distances")
    a.plot([1], [R[none]["dmin"]], "s", color=GREEN, label="No coupling")
    a.plot(np.full(len(perms), 2), [R[k]["dmin"] for k in perms], "^", color=RED, label="Permuted distances")
    a.set_xticks([0, 1, 2]); a.set_xticklabels(["Real", "None", "Permuted"]); a.set_title(title)
ax[0].set_ylabel("Minimum binomial deviance"); ax[0].legend(fontsize=8, frameon=False)
fig.tight_layout(); save(fig, "fig16_test_placebo_en.png")

# ---- fig17: identifiability (weighted histograms vs prior)
r = R["ident/orig"]; th = np.array(r["thetas"]); w = np.array(r["weights"])
names = abc_smc.NAMES
fig, ax = plt.subplots(2, 3, figsize=(10, 5.6))
for a, j in zip(ax.ravel(), (1, 2, 4, 5, 3, 6)):
    lo_, hi_ = abc_smc.BOUNDS[j]
    if abc_smc.LOG[j]:
        x = np.log10(th[:, j]); edges = np.linspace(np.log10(lo_), np.log10(hi_), 25)
        a.hist(x, bins=edges, weights=w, density=True, color=BLUE, alpha=0.7)
        a.plot([edges[0], edges[-1]], [1 / (edges[-1] - edges[0])] * 2, "k--", lw=1)
        a.set_xlabel(f"log10({names[j]})")
    else:
        edges = np.linspace(lo_, hi_, 25)
        a.hist(th[:, j], bins=edges, weights=w, density=True, color=BLUE, alpha=0.7)
        a.plot([lo_, hi_], [1 / (hi_ - lo_)] * 2, "k--", lw=1)
        a.set_xlabel(names[j])
    a.set_yticks([])
fig.tight_layout(); save(fig, "fig17_identifiabilite_en.png")

# ---- fig18: ABC-SMC convergence
h = R["struct/MSEIR/full/g7/s1"]["history"]
fig, a = plt.subplots(figsize=(6.5, 3.8))
a.plot([e["gen"] for e in h], [e["median"] for e in h], "-o", color=BLUE, label="Median deviance of the population")
a.plot([e["gen"] for e in h], [e["min"] for e in h], "-s", color=RED, label="Best particle")
a.set_yscale("log"); a.set_xlabel("Generation"); a.set_ylabel("Binomial deviance"); a.set_xticks([e["gen"] for e in h])
a.legend(frameon=False, fontsize=8)
fig.tight_layout(); save(fig, "fig18_convergence_en.png")
print("figures written to", A.out)
