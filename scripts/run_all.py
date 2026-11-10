"""Re-run every calibration reported in the manuscript and save the results to results/.

usage:  python scripts/run_all.py --ndvi <NDVI csv> --ndwi <NDWI csv> --coords <ponds csv> [--out results]
Each job is a recalibration by ABC-SMC (a few seconds each). Seeds 1 and 2 are the two seeds.
"""
import argparse, json, sys, os, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fvr.data import distance_matrix
from fvr.forcing import build_forcing
from fvr.experiments import calibrate, permute_distances
from fvr import abc_smc

ARGS = None
G7, G5 = dict(gamma=1 / 7.0), dict(gamma=1 / 5.0)
STRUCT = {"MSEIR": {}, "SIR": dict(use_M=False, use_E=False), "MSIR": dict(use_E=False), "SEIR": dict(use_M=False)}
SPATIAL = {"full": {}, "none": dict(spatial="none"), "distance": dict(spatial="distance"), "background": dict(spatial="background")}
SETS = {"ref": dict(), "A": dict(life_years=2, mu_v=1 / 7, delta=1 / 5), "B": dict(life_years=2, mu_v=1 / 21, delta=1 / 14),
        "C": dict(life_years=6, mu_v=1 / 7, delta=1 / 14), "D": dict(life_years=6, mu_v=1 / 21, delta=1 / 5),
        "literature": dict(life_years=4, mu_v=0.08, delta=1 / 10.5)}
WIDE = np.array([(0.0, 10.0), (1e-4, 2.0), (0.01, 2000.0), (1 / 400, 1 / 2), (1e-8, 1.0), (0.01, 10.0), (1e-9, 0.1)])
LOWN = abc_smc.BOUNDS.copy(); LOWN[2, 0] = 0.01


def jobs():
    J = []
    for g, gk in (("g7", G7), ("g5", G5)):
        for sname, s in STRUCT.items():
            for seed in (1, 2):
                J.append((f"struct/{sname}/full/{g}/s{seed}", dict(seed=seed, fixed=gk, **s)))
        for sp in ("none", "distance", "background"):
            for seed in (1, 2):
                J.append((f"spatial/MSEIR/{sp}/{g}/s{seed}", dict(seed=seed, fixed=gk, **SPATIAL[sp])))
    for i in range(5):   # placebo: permuted distances, distance-only model
        J.append((f"placebo/ndvi/perm{i}", dict(seed=10 + i, fixed=G7, spatial="distance", perm=100 + i)))
    for i in range(5):
        J.append((f"placebo/ndvi_full/perm{i}", dict(seed=20 + i, fixed=G7, perm=200 + i)))
    for sname, s in (("MSEIR", {}), ("SIR", STRUCT["SIR"])):
        J.append((f"ndwi/{sname}/full", dict(seed=1, fixed=G7, forcing="NDWI", **s)))
    for sp in ("none", "distance"):
        for seed in (1, 2):
            J.append((f"ndwi/MSEIR/{sp}/s{seed}", dict(seed=seed, fixed=G7, forcing="NDWI", **SPATIAL[sp])))
    for i in range(3):
        J.append((f"ndwi/placebo/perm{i}", dict(seed=30 + i, fixed=G7, forcing="NDWI", spatial="distance", perm=300 + i)))
    lat = dict(epsilon=1 / 1.5)
    for sname in ("MSEIR", "SEIR", "SIR"):
        J.append((f"latency1.5/{sname}/full", dict(seed=1, fixed=lat, **STRUCT[sname])))
    J.append(("latency1.5/MSEIR/none", dict(seed=1, fixed=lat, spatial="none")))
    J.append(("countM/MSEIR/full", dict(seed=1, fixed=G7, count_M=True)))
    for name, fx in SETS.items():
        for mname, kw in (("MSEIR_full", {}), ("SIR_full", STRUCT["SIR"]), ("MSEIR_none", dict(spatial="none"))):
            J.append((f"sens/{name}/{mname}", dict(seed=1, fixed=fx, **kw)))
    # Introduction spread over all six age classes. Definition 1 (default): the same infection
    # pressure Intro(t) * S_a / N is applied to every class. Definitions 2 and 3 keep the baseline
    # total and split it by susceptible share / equally. Each is run with the default prior bound
    # n_intro >= 5 and with the bound lowered to 0.01.
    for mode in (1, 2, 3):
        for seed in (1, 2):
            J.append((f"nomad/allages_def{mode}/s{seed}", dict(seed=seed, fixed=G7, nomad_all_ages=mode)))
            J.append((f"nomad/allages_def{mode}_lown/s{seed}", dict(seed=seed, fixed=G7, nomad_all_ages=mode, bounds=LOWN)))
    for seed in (1, 2):
        J.append((f"nomad/concentrated_lown/s{seed}", dict(seed=seed, fixed=G7, bounds=LOWN)))
    for pe in (0.02, 0.6, 0.88):
        J.append((f"nomad/pearly{pe}", dict(seed=1, fixed=dict(p_early=pe))))
    J.append(("ident/orig", dict(seed=1, fixed=G7, beta_direct_fixed=4.8, abc_kwargs=dict(N0=1000, N=1000))))
    J.append(("ident/lown", dict(seed=1, fixed=G7, beta_direct_fixed=4.8, bounds=LOWN, abc_kwargs=dict(N0=1000, N=1000))))
    J.append(("ident/wide_literature", dict(seed=1, fixed=SETS["literature"], beta_direct_fixed=4.8, bounds=WIDE,
                                           abc_kwargs=dict(N0=1000, N=1000))))
    return J


def load():
    D, ids = distance_matrix(ARGS.coords)
    F = {"NDVI": build_forcing(ARGS.ndvi, "NDVI"), "NDWI": build_forcing(ARGS.ndwi, "NDWI")}
    return D, F


def work(job):
    name, kw = job
    D, F = load()
    kw = dict(kw)
    Ed, season, _ = F[kw.pop("forcing", "NDVI")]
    perm = kw.pop("perm", None)
    Dm = permute_distances(D, np.random.default_rng(perm)) if perm is not None else D
    t = time.time()
    r = calibrate(Ed, season, Dm, **kw)
    out = dict(name=name, dmin=r["dmin"], k=r["k"], n_gen=r["n_gen"], n_final=len(r["dev"]),
               theta_best=r["theta_best"].tolist(), history=r["history"],
               thetas=r["thetas"].tolist(), weights=r["weights"].tolist(), seconds=time.time() - t)
    print(f"{name:40s} Dmin={r['dmin']:.2f} k={r['k']} gens={r['n_gen']} ({out['seconds']:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndvi", required=True); ap.add_argument("--ndwi", required=True)
    ap.add_argument("--coords", required=True); ap.add_argument("--out", default="results")
    ap.add_argument("--procs", type=int, default=2)
    ARGS = ap.parse_args()
    os.makedirs(ARGS.out, exist_ok=True)
    with Pool(ARGS.procs) as pool:
        res = pool.map(work, jobs(), chunksize=1)
    json.dump({r["name"]: r for r in res}, open(os.path.join(ARGS.out, "results.json"), "w"))
    print("saved", os.path.join(ARGS.out, "results.json"))
