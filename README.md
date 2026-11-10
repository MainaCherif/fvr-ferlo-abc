# fvr-ferlo-abc

Code of the manuscript *Spatial modelling of Rift Valley fever transmission in the Ferlo, Senegal: what
age-specific seroprevalence data can and cannot tell us* (M. Kane, C. Diallo, D. Mboup).

Spatial M-S-E-I-R (hosts, 6 annual age classes) / S-L-I (vectors) model on 14 temporary ponds, forced by
pond-specific MODIS NDVI (NDWI as a sensitivity test), calibrated by ABC-SMC on six age-specific IgG
seroprevalences (Durand et al. 2020, Table 4) with the binomial deviance.

## Layout
- `fvr/model.py` - the model (daily Euler scheme, numba). Eqs. for M, S, E, I, R, S_v, L_v, I_v, the force of infection
  (vector-borne + distance-dependent + constant background + direct) and the seasonal nomadic introduction.
- `fvr/abc_smc.py` - ABC-SMC (350 Latin-hypercube draws, 30 % kept, 300 proposals per generation, acceptance below the median
  deviance of the previous generation, stop when the median changes by < 2 %).
- `fvr/experiments.py` - model variants (structures, spatial coupling, permuted distances, introduction on all age classes ...).
- `fvr/data.py`, `fvr/forcing.py` - observed counts, catalytic reference, pond distances, environmental forcing.
- `fvr/analysis.py` - quantities derived from the final populations (WAIC, deviance profiles).
- `scripts/run_all.py` - reruns every calibration of the manuscript (about 5 min on 2 cores); `scripts/report.py` - prints every number of the Results; `scripts/make_figures.py` - redraws the result figures from `results.json` (not included in the archive; run the script).
- `results/results.json` - output of `run_all.py` (best deviance, final populations, history of every calibration).
- `results/summary_for_manuscript.txt` - output of `report.py`: the numbers produced by this code for each analysis of the Results.
- `results/comparison_with_manuscript.txt` - these numbers next to the values printed in the manuscript.
- `data/seroprevalence_durand2020_table4.csv` - calibration target (published in Durand et al. 2020).

## Data that are not in this repository
The coordinates of the 14 ponds were provided by B. Durand (personal communication) and are not redistributed here.
The AppEEARS exports contain these coordinates, so they are not redistributed either. To rerun, put in `data_restricted/`:
`ponds_coordinates.csv` (columns `ID_mare`, `Latitude_N`, `Longitude_E`), `appeears_ndvi.csv` and `appeears_ndwi.csv`
(MOD13Q1 v061, NDVI, NIR and MIR reflectance, pixel reliability; 2007-12-19 to 2015-12-19, from https://appeears.earthdatacloud.nasa.gov).
Contact the authors for the coordinates, subject to B. Durand's agreement.

## Run
```
pip install -r requirements.txt
python scripts/run_all.py --ndvi data_restricted/appeears_ndvi.csv --ndwi data_restricted/appeears_ndwi.csv \
       --coords data_restricted/ponds_coordinates.csv --out results
python scripts/report.py results/results.json --ndvi data_restricted/appeears_ndvi.csv --coords data_restricted/ponds_coordinates.csv
python scripts/make_figures.py --ndvi data_restricted/appeears_ndvi.csv --coords data_restricted/ponds_coordinates.csv
python -m pytest tests
```
Tested with Python 3.13, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, numba 0.68.1. Seeds are fixed (seeds 1 and 2 are the two seeds
of the manuscript), so a rerun gives the same numbers: a complete rerun was checked to reproduce `results/results.json` exactly (same deviances and same final populations).

## Agreement with the manuscript
This repository implements the model and the ABC-SMC calibration described in the manuscript. ABC-SMC is stochastic, so
the deviances agree with those of the manuscript to within the Monte Carlo variability of the calibration (about 0.3 point
between seeds), not to the last digit: all variants stay between 177.8 and 178.6, as in the manuscript, and every conclusion
on the model structures, the spatial coupling (permutation test) and the NDVI/NDWI comparison is reproduced.
Two results differ and should be read from `results/comparison_with_manuscript.txt`: the deviance obtained when the introduction is
spread over all age classes depends on the lower prior bound of `n_intro` (264 with the bound 5, about 181 with 0.01), and
the two-vector-species variant is not implemented here.

## License
MIT (code). Cite the manuscript and Durand et al. (2020) for the data.
