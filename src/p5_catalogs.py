"""
Stage 5: Build the comparison data - event catalogs -> per-hour ground-truth labels.

ICME events   : HELIO4CAST ICMECAT v2.3 (Moestl et al.), Wind rows only (Wind sits at L1 like OMNI).
                An ICME interval runs from icme_start_time to mo_end_time.
                File: data/catalogs/HELIO4CAST_ICMECAT_v23.csv
CIR/HSS events: by default the Wind high-speed-stream (HSS) intervals in the HELIO4CAST SIRCAT v1.0
                (taken from Grandin et al. 2019). HSS intervals are the fast-stream part of a CIR
                structure, so this is a PROXY for the Hajra & Sunny (2022) CIR list, not the same
                list. File: data/catalogs/HELIO4CAST_SIRCAT_v10_Wind.csv
                To use the real CIR list instead, add data/catalogs/cir_catalog.csv with columns
                `start,end` (optional `id`); it then replaces the SIRCAT intervals.

Outputs: outputs/event_labels.csv (per-hour 0 = quiet, 1 = ICME, 2 = CIR/HSS) and
         outputs/catalog_events.csv (the catalog intervals actually used)
"""
import os

import numpy as np
import pandas as pd

from src.compare_utils import EvalContext, OUT_DIR, hour_timestamps, reset_scores

CATALOG_DIR = os.path.join("data", "catalogs")
ICMECAT_PATH = os.path.join(CATALOG_DIR, "HELIO4CAST_ICMECAT_v23.csv")
SIRCAT_PATH = os.path.join(CATALOG_DIR, "HELIO4CAST_SIRCAT_v10_Wind.csv")
CIR_PATH = os.path.join(CATALOG_DIR, "cir_catalog.csv")
FEATURES_PATH = os.path.join(OUT_DIR, "processed_omni_features.csv")

_CACHE = {}


def _utc_naive(series):
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)


def load_icme_events(t0, t1):
    cat = pd.read_csv(ICMECAT_PATH)
    cat = cat[cat["sc_insitu"].astype(str).str.lower() == "wind"].copy()
    start, end = _utc_naive(cat["icme_start_time"]), _utc_naive(cat["mo_end_time"])
    end = end.fillna(start + pd.to_timedelta(cat["icme_duration"], unit="h"))
    ev = pd.DataFrame({"type": "ICME", "id": cat["icmecat_id"].values,
                       "start": start.values, "end": end.values}).dropna(subset=["start", "end"])
    return ev[(ev["end"] >= t0) & (ev["start"] <= t1)].sort_values("start").reset_index(drop=True)


def load_cir_events(t0, t1):
    """User-supplied CIR list if present, otherwise SIRCAT Wind HSS intervals."""
    if os.path.exists(CIR_PATH):
        raw = pd.read_csv(CIR_PATH)
        low = {c.lower().strip(): c for c in raw.columns}
        s = next((low[k] for k in ("start", "start_time", "cir_start") if k in low), None)
        e = next((low[k] for k in ("end", "end_time", "cir_end") if k in low), None)
        if s and e:
            ev = pd.DataFrame({
                "type": "CIR",
                "id": raw[low["id"]] if "id" in low else [f"CIR_{i}" for i in range(len(raw))],
                "start": pd.to_datetime(raw[s], errors="coerce"),
                "end": pd.to_datetime(raw[e], errors="coerce")}).dropna(subset=["start", "end"])
            src = f"user-supplied {CIR_PATH}"
        else:
            raise ValueError(f"{CIR_PATH} needs 'start' and 'end' columns")
    elif os.path.exists(SIRCAT_PATH):
        cat = pd.read_csv(SIRCAT_PATH)
        cat = cat[cat["sc_insitu"].astype(str).str.lower() == "wind"]
        ev = pd.DataFrame({"type": "CIR", "id": cat["sircat_id"].values,
                           "start": _utc_naive(cat["hss_start_time"]).values,
                           "end": _utc_naive(cat["hss_end_time"]).values}
                          ).dropna(subset=["start", "end"])
        src = "SIRCAT v1.0 Wind high-speed streams (Grandin et al. 2019) - CIR proxy"
    else:
        return pd.DataFrame(columns=["type", "id", "start", "end"]), "none"
    ev = ev[(ev["end"] >= t0) & (ev["start"] <= t1)].sort_values("start").reset_index(drop=True)
    return ev, src


def get_context(verbose=False):
    """Build (and cache) the EvalContext; writes the label / event CSVs on first call."""
    if "ctx" in _CACHE:
        return _CACHE["ctx"]
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"{FEATURES_PATH} not found - run the base pipeline first")
    time_df = pd.read_csv(FEATURES_PATH, usecols=["YEAR", "DOY", "HR"])
    times = hour_timestamps(time_df)
    t0, t1 = times.iloc[0], times.iloc[-1] + pd.Timedelta(hours=1)

    cir, cir_src = load_cir_events(t0, t1)
    events = pd.concat([load_icme_events(t0, t1), cir], ignore_index=True)
    ctx = EvalContext(times, events, cir_source=cir_src)

    labels = time_df.copy()
    labels["TIMESTAMP"] = times.values
    labels["ICME"] = ctx.mask["ICME"].astype(int)
    labels["CIR_HSS"] = ctx.mask["CIR"].astype(int)
    labels["LABEL"] = ctx.label
    os.makedirs(OUT_DIR, exist_ok=True)
    labels.to_csv(os.path.join(OUT_DIR, "event_labels.csv"), index=False)
    events.to_csv(os.path.join(OUT_DIR, "catalog_events.csv"), index=False)

    if verbose:
        n_i, n_c = len(ctx.runs["ICME"]), len(ctx.runs["CIR"])
        print(f"    ICME events   : {n_i} ({int(ctx.mask['ICME'].sum())} hours, "
              f"{100 * ctx.mask['ICME'].mean():.1f}% of data)  [HELIO4CAST ICMECAT v2.3, Wind]")
        print(f"    CIR/HSS events: {n_c} ({int(ctx.mask['CIR'].sum())} hours, "
              f"{100 * ctx.mask['CIR'].mean():.1f}% of data)  [{cir_src}]")
        print(f"    Quiet hours   : {int((ctx.label == 0).sum())} "
              f"({100 * (ctx.label == 0).mean():.1f}%)")
        print(f"    Split: train before {TEST_START_STR}; test from then on "
              f"(test holds {sum(1 for r in ctx.runs['ICME'] if r[0] >= ctx.test_lo)} ICME / "
              f"{sum(1 for r in ctx.runs['CIR'] if r[0] >= ctx.test_lo)} CIR/HSS events)")
    _CACHE["ctx"] = ctx
    return ctx


TEST_START_STR = "2013-07-01"


def run_catalog_labeling():
    print("Catalogs: building per-hour ICME / CIR-HSS ground-truth labels...")
    reset_scores()
    _CACHE.clear()
    get_context(verbose=True)
    print("    Saved outputs/event_labels.csv and outputs/catalog_events.csv")


if __name__ == "__main__":
    run_catalog_labeling()