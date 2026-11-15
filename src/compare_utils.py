"""
Shared helpers for the catalog-comparison stages (p5-p8).

  * EvalContext - catalog ground truth aligned to the OMNI hourly index, with hour-level and
    EVENT-level precision / recall / F1 (the proposal asks for event-level scoring because
    disturbances are rare and point-by-point accuracy is misleading)
  * best_cluster / majority_map - how an unsupervised cluster is turned into a detector or a class
    label WITHOUT looking at the test period
  * record_score - appends a row to outputs/method_scores.csv for the final comparison tables

Assumptions (state them in the report):
  * A "predicted event" is a run of flagged hours; gaps of <= MAX_GAP_H unflagged hours are bridged
    and runs shorter than MIN_RUN_H hours are dropped.
  * A predicted and a catalog event "match" when their temporal intersection-over-union is
    >= IOU_MIN. Plain "any overlap" is too lenient: one run covering the whole dataset would score a
    perfect F1.
  * Event precision = share of predicted events that match a catalog event.
    Event recall    = share of catalog events matched by a predicted event.
  * Supervised models train on hours before TEST_START and are scored on hours from TEST_START on,
    with a SPLIT_GAP_H-hour gap so the rolling windows cannot leak. Unsupervised methods are scored
    on the full period and on the same test period; cluster choices use the training period only.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

OUT_DIR = "outputs"
SCORES_PATH = os.path.join(OUT_DIR, "method_scores.csv")

TEST_START = pd.Timestamp("2013-07-01")
SPLIT_GAP_H = 48
MIN_RUN_H = 3
MAX_GAP_H = 3
IOU_MIN = 0.1

TIME_COLS = ["YEAR", "DOY", "HR"]
TARGETS = ("ICME", "CIR")                     # catalog classes (label 1 = ICME, 2 = CIR/HSS)
TARGET_NAMES = {"ICME": "ICME", "CIR": "CIR/HSS"}
CLASS_OF = {"ICME": 1, "CIR": 2}
# NASA OMNI hourly fill codes (these are NOT masked by src/p1_preprocess.py)
FILL_THRESHOLDS = {"FLOW_SPEED": 9999.0, "TEMP": 9999999.0, "PLASMA_BETA": 999.99}


# ----------------------------------------------------------------------------- time / fill helpers
def hour_timestamps(df):
    """Start-of-hour timestamps from YEAR / DOY / HR columns."""
    base = pd.to_datetime(df["YEAR"].astype(int).astype(str) + "-01-01")
    return (base + pd.to_timedelta(df["DOY"].astype(int) - 1, unit="D")
            + pd.to_timedelta(df["HR"].astype(int), unit="h"))


def fill_flag(df):
    """True for hours whose raw columns hold an OMNI fill code."""
    flag = np.zeros(len(df), dtype=bool)
    for col, thr in FILL_THRESHOLDS.items():
        if col in df.columns:
            flag |= df[col].to_numpy() >= thr * 0.99
    return flag


def runs_from_mask(mask, min_len=MIN_RUN_H, max_gap=MAX_GAP_H):
    """Contiguous runs (start_idx, end_idx inclusive) of a boolean mask; gaps <= max_gap are
    bridged and runs shorter than min_len are dropped."""
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev - 1 <= max_gap:
            prev = i
        else:
            runs.append((int(start), int(prev)))
            start = prev = i
    runs.append((int(start), int(prev)))
    return [(a, b) for a, b in runs if b - a + 1 >= min_len]


# ----------------------------------------------------------------------------- event metrics
def _overlaps_any(run, others):
    a, b = run
    return any(a <= d and c <= b for c, d in others)


def _iou(r1, r2):
    inter = max(0, min(r1[1], r2[1]) - max(r1[0], r2[0]) + 1)
    union = (r1[1] - r1[0] + 1) + (r2[1] - r2[0] + 1) - inter
    return inter / union


def _matches(run, others):
    return any(_iou(run, o) >= IOU_MIN for o in others)


def event_prf1(pred_runs, cat_runs):
    n_pred, n_cat = len(pred_runs), len(cat_runs)
    tp_pred = sum(_matches(r, cat_runs) for r in pred_runs)
    tp_cat = sum(_matches(r, pred_runs) for r in cat_runs)
    prec = tp_pred / n_pred if n_pred else 0.0
    rec = tp_cat / n_cat if n_cat else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"event_precision": prec, "event_recall": rec, "event_f1": f1,
            "n_pred_events": n_pred, "n_catalog_events": n_cat}


class EvalContext:
    """Ground truth from the event catalogs, aligned to the OMNI hourly index."""

    def __init__(self, times, events, cir_source=""):
        self.times = pd.DatetimeIndex(times)
        self.n = len(self.times)
        self.events = events
        self.cir_source = cir_source
        self.mask, self.runs = {}, {}
        for kind in TARGETS:
            ev = events[events["type"] == kind]
            m = np.zeros(self.n, dtype=bool)
            runs = []
            for s, e in zip(ev["start"], ev["end"]):
                i0 = int(self.times.searchsorted(s - pd.Timedelta(hours=1), side="right"))
                i1 = int(self.times.searchsorted(e, side="left")) - 1
                if i1 >= i0:
                    m[i0:i1 + 1] = True
                    runs.append((i0, i1))
            self.mask[kind], self.runs[kind] = m, runs
        # 0 = quiet, 1 = ICME, 2 = CIR/HSS (ICME wins where they overlap)
        self.label = np.where(self.mask["ICME"], 1, np.where(self.mask["CIR"], 2, 0))
        self.test_lo = int(self.times.searchsorted(TEST_START))

    def has(self, kind):
        return len(self.runs[kind]) > 0

    def kinds(self):
        return [k for k in TARGETS if self.has(k)]

    def bounds(self, scope):
        """(lo, hi) hour-index bounds for 'full', 'train' or 'test'."""
        if scope == "full":
            return 0, self.n
        if scope == "test":
            return self.test_lo, self.n
        if scope == "train":
            return 0, max(0, self.test_lo - SPLIT_GAP_H)
        raise ValueError(scope)

    def split_names(self):
        s = np.full(self.n, "train", dtype=object)
        s[max(0, self.test_lo - SPLIT_GAP_H):self.test_lo] = "gap"
        s[self.test_lo:] = "test"
        return s

    def evaluate(self, pred_mask, kind="ICME", scope="full"):
        """Hour-level and event-level scores of a boolean per-hour prediction."""
        lo, hi = self.bounds(scope)
        pred = np.asarray(pred_mask, dtype=bool)
        hp, hr, hf, _ = precision_recall_fscore_support(
            self.mask[kind][lo:hi], pred[lo:hi], average="binary", zero_division=0)
        pred_runs = [(a + lo, b + lo) for a, b in runs_from_mask(pred[lo:hi])]
        cat_runs = [r for r in self.runs[kind] if lo <= (r[0] + r[1]) / 2 < hi]
        out = event_prf1(pred_runs, cat_runs)
        base = float(self.mask[kind][lo:hi].mean())
        out.update({"hour_precision": hp, "hour_recall": hr, "hour_f1": hf,
                    "hour_lift": hp / base if base else float("nan")})   # 1.0 = no better than chance
        return out

    def run_overlaps_catalog(self, run):
        return any(_overlaps_any(run, self.runs[k]) for k in TARGETS)


# ----------------------------------------------------------------------------- cluster handling
def best_cluster(ctx, labels, kind):
    """Cluster that best detects `kind`, chosen on the TRAINING period only (hour-level F1, then
    event F1) so the test period never influences the choice. Hour-level F1 is used first because
    the event metric can flatter a cluster that simply covers most of the data."""
    labels = np.asarray(labels)
    best, best_key = None, (-1.0, -1.0)
    for c in sorted(set(labels.tolist())):
        m = ctx.evaluate(labels == c, kind, "train")
        key = (m["hour_f1"], m["event_f1"])
        if key > best_key:
            best, best_key = c, key
    return best


def enrichment_map(ctx, labels):
    """{cluster: catalog class (0/1/2)}: each cluster gets the class it is most over-represented in
    (share inside the cluster / class base rate), using the TRAINING period only. A plain majority
    vote would label an ICME-enriched cluster 'quiet' because ICMEs are rare overall."""
    lo, hi = ctx.bounds("train")
    labels = np.asarray(labels)[lo:hi]
    truth = ctx.label[lo:hi]
    base = np.bincount(truth, minlength=3) / len(truth)
    mapping = {}
    for c in sorted(set(labels.tolist())):
        sel = labels == c
        if not sel.any():
            mapping[c] = 0
            continue
        share = np.bincount(truth[sel], minlength=3) / sel.sum()
        lift = np.where(base > 0, share / np.where(base > 0, base, 1), 0)
        mapping[c] = int(lift.argmax())
    return mapping


# ----------------------------------------------------------------------------- score log
def record_score(method, target, scope, metrics):
    """Upsert one row (method, target, scope) in outputs/method_scores.csv."""
    os.makedirs(OUT_DIR, exist_ok=True)
    row = {"method": method, "target": TARGET_NAMES.get(target, target), "scope": scope}
    row.update({k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                for k, v in metrics.items()})
    if os.path.exists(SCORES_PATH):
        df = pd.read_csv(SCORES_PATH)
        keep = ~((df["method"] == row["method"]) & (df["target"] == row["target"])
                 & (df["scope"] == scope))
        df = pd.concat([df[keep], pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(SCORES_PATH, index=False)


def reset_scores():
    if os.path.exists(SCORES_PATH):
        os.remove(SCORES_PATH)