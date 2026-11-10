"""Pond-specific environmental forcing E_d^i(t) from MODIS MOD13Q1 v061 (AppEEARS export)."""
import numpy as np
import pandas as pd

START, END = "2008-01-01", "2015-12-31"   # 2922 days = 8 years


def build_forcing(csv_path, index="NDVI"):
    """Return (Ed [n_ponds, n_days] in [0,1], season [n_days], pond ids).

    Observations with pixel reliability 2 or 3 are discarded, each series is linearly
    interpolated to a daily series, then min-max normalised with a single (NDVI_min,
    NDVI_max) common to all ponds and the whole period (Eq. 2).
    season: 0 dry, 1 first half of the rainy season (Jul-Aug), 2 second half (Sep-Oct).
    """
    df = pd.read_csv(csv_path)
    rel = "MOD13Q1_061__250m_16_days_pixel_reliability"
    if index == "NDVI":
        df["val"] = df["MOD13Q1_061__250m_16_days_NDVI"]
    else:  # NDWI = (NIR - MIR) / (NIR + MIR)
        nir = df["MOD13Q1_061__250m_16_days_NIR_reflectance"]
        mir = df["MOD13Q1_061__250m_16_days_MIR_reflectance"]
        df["val"] = (nir - mir) / (nir + mir)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df[rel].isin([0, 1])]
    days = pd.date_range(START, END, freq="D")
    ids = sorted(df["ID"].unique())
    series = []
    for pid in ids:
        s = df[df["ID"] == pid].sort_values("Date")
        t = (s["Date"] - pd.Timestamp(START)).dt.days.values
        series.append(np.interp(np.arange(len(days)), t, s["val"].values))
    series = np.array(series)
    vmin = df["val"].min(); vmax = df["val"].max()
    Ed = (series - vmin) / (vmax - vmin)
    month = days.month.values
    season = np.where(np.isin(month, [7, 8]), 1, np.where(np.isin(month, [9, 10]), 2, 0)).astype(np.int64)
    return Ed, season, ids
