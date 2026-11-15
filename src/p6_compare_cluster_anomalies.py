"""
Stage 6 - Comparison 1: do the k-Means clusters and Isolation Forest anomalies line up with the
catalogued ICME and CIR/HSS intervals?  (Paper: "Compare the cluster/anomaly intervals against the
Richardson & Cane + Hajra & Sunny catalogs".)

Uses the report's own results: KMEANS_CLUSTER from outputs/omni_clustered_results.csv (k = 5) and the
flags/scores in outputs/omni_anomaly_results.csv (contamination = 0.05), so nothing is re-fit except
the contamination sweep.

  * Matching matrix: for each cluster, the % of its hours inside a catalog ICME / CIR-HSS / neither,
    plus the % of hours that hold an OMNI fill code and the lift over the catalog base rate.
  * Cluster-as-detector: each cluster scored as an ICME (and CIR/HSS) detector. The cluster used for
    the comparison table is the one with the best TRAINING-period event F1.
  * Cluster x anomaly cross-check (tests the report's statement about clusters 2 and 4 and the
    anomaly scores).
  * Isolation Forest at the report setting, scored against each catalog (hour and event level, plus
    threshold-free average precision / ROC-AUC), a contamination sweep, and a list of strong
    anomaly runs that match no catalog event (candidates for manual inspection).
  * Overlay plots: |B|, speed, beta with clusters and anomalies under shaded catalog intervals.

Outputs: outputs/matching_matrix_kmeans.csv/.png, cluster_detector_scores.csv,
         cluster_vs_anomaly.csv, if_contamination_sweep.csv/.png, uncataloged_anomalies.csv,
         overlay_*.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.compare_utils import (OUT_DIR, TARGET_NAMES, TIME_COLS, best_cluster, fill_flag,
                               hour_timestamps, record_score, runs_from_mask)
from src.p5_catalogs import get_context

REPORT_CONTAMINATION = 0.05          # value used in src/p3_anomaly.py
CONTAMINATIONS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
TOP_N_UNCATALOGED = 25
SHADE = {"ICME": "tab:orange", "CIR": "tab:green"}
METRIC_KEYS = ["event_precision", "event_recall", "event_f1", "hour_precision", "hour_recall",
               "hour_f1", "n_pred_events", "n_catalog_events"]


# ----------------------------------------------------------------------------- matching matrix
def matching_matrix(labels, ctx, fill_rows):
    df = pd.DataFrame({"cluster": labels, "ICME": ctx.mask["ICME"], "CIR": ctx.mask["CIR"],
                       "fill": fill_rows})
    df["quiet"] = ~(df.ICME | df.CIR)
    g = df.groupby("cluster")
    mm = pd.DataFrame({"hours": g.size(),
                       "pct_ICME": 100 * g["ICME"].mean(),
                       "pct_CIR_HSS": 100 * g["CIR"].mean(),
                       "pct_quiet": 100 * g["quiet"].mean(),
                       "pct_fill_rows": 100 * g["fill"].mean()})
    for kind, col in (("ICME", "ICME_lift"), ("CIR", "CIR_HSS_lift")):
        base = 100 * df[kind].mean()
        mm[col] = mm[f"pct_{'ICME' if kind == 'ICME' else 'CIR_HSS'}"] / base if base else np.nan
    mm = mm.round(2)
    mm.to_csv(os.path.join(OUT_DIR, "matching_matrix_kmeans.csv"))

    fig, ax = plt.subplots(figsize=(8, 0.7 * len(mm) + 2))
    sns.heatmap(mm[["pct_ICME", "pct_CIR_HSS", "pct_quiet", "pct_fill_rows"]], annot=True,
                fmt=".1f", cmap="viridis", cbar_kws={"label": "% of cluster's hours"}, ax=ax)
    ax.set_title("Matching matrix: k-Means clusters vs. catalogs\n"
                 f"(base rates: ICME {100 * df.ICME.mean():.1f}%, "
                 f"CIR/HSS {100 * df.CIR.mean():.1f}%)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "matching_matrix_kmeans.png"), dpi=200)
    plt.close(fig)
    return mm


# ----------------------------------------------------------------------------- overlay plots
def _plot_window(df, times, cluster, anomaly, ctx, t0, t1, path, title):
    sel = (times >= t0) & (times < t1)
    t, d = times[sel], df[sel]
    cl, an = cluster[sel], anomaly[sel].astype(int)
    bad = fill_flag(d)
    v = np.where(bad, np.nan, d["FLOW_SPEED"].to_numpy())
    beta = np.where(bad, np.nan, d["PLASMA_BETA"].to_numpy())

    fig, axes = plt.subplots(5, 1, figsize=(13, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 3, 0.6, 0.6]})
    axes[0].plot(t, d["ABS_B"], lw=0.8, color="k"); axes[0].set_ylabel("|B| (nT)")
    axes[1].plot(t, v, lw=0.8, color="k"); axes[1].set_ylabel("Speed (km/s)")
    axes[2].semilogy(t, np.clip(beta, 1e-2, None), lw=0.8, color="k")
    axes[2].set_ylabel("Plasma beta")
    x0, x1 = mdates.date2num(t.iloc[0]), mdates.date2num(t.iloc[-1])
    axes[3].imshow(cl[np.newaxis, :], aspect="auto", cmap="tab10", vmin=0, vmax=9,
                   extent=[x0, x1, 0, 1], interpolation="nearest")
    axes[3].set_yticks([]); axes[3].set_ylabel("k-Means", rotation=0, ha="right", va="center")
    axes[4].imshow(an[np.newaxis, :], aspect="auto", cmap="Reds", vmin=0, vmax=1.3,
                   extent=[x0, x1, 0, 1], interpolation="nearest")
    axes[4].set_yticks([]); axes[4].set_ylabel("Anomaly", rotation=0, ha="right", va="center")
    for kind in ("ICME", "CIR"):
        sub = ctx.events[ctx.events.type == kind]
        for s, e in zip(sub["start"], sub["end"]):
            if e >= t.iloc[0] and s <= t.iloc[-1]:
                for ax in axes[:3]:
                    ax.axvspan(s, e, color=SHADE[kind], alpha=0.25, lw=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4, label=TARGET_NAMES[k])
               for k, c in SHADE.items() if ctx.has(k)]
    axes[0].legend(handles=handles, loc="upper right")
    axes[0].set_xlim(x0, x1)          # shaded intervals that start earlier must not stretch the axis
    axes[0].set_title(title)
    axes[4].xaxis_date(); axes[4].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _overlay_plots(ctx, df, cluster, anomaly):
    times = hour_timestamps(df)
    starts = ctx.events["start"]
    cands = pd.date_range(times.iloc[0].normalize(), times.iloc[-1] - pd.Timedelta(days=60),
                          freq="10D")
    best = max(cands, key=lambda c: ((starts >= c) & (starts < c + pd.Timedelta(days=60))).sum())
    _plot_window(df, times, cluster, anomaly, ctx, best, best + pd.Timedelta(days=60),
                 os.path.join(OUT_DIR, f"overlay_overview_{best:%Y%m%d}.png"),
                 f"60-day overview from {best:%Y-%m-%d}: clusters and anomalies vs. catalog events")
    for kind in ctx.kinds():
        ev = ctx.events[ctx.events.type == kind].copy()
        row = ev.assign(dur=ev["end"] - ev["start"]).sort_values("dur", ascending=False).iloc[0]
        _plot_window(df, times, cluster, anomaly, ctx, row["start"] - pd.Timedelta(days=2),
                     row["end"] + pd.Timedelta(days=2),
                     os.path.join(OUT_DIR, f"overlay_{kind.lower()}_{row['start']:%Y%m%d}.png"),
                     f"Longest catalogued {TARGET_NAMES[kind]} event ({row['id']})")


# ----------------------------------------------------------------------------- main stage
def run_cluster_anomaly_comparison():
    print("Comparison 1: clusters and anomalies vs. the event catalogs...")
    ctx = get_context()
    df = pd.read_csv(os.path.join(OUT_DIR, "processed_omni_features.csv"))
    fill_rows = fill_flag(df)
    cluster = pd.read_csv(os.path.join(OUT_DIR, "omni_clustered_results.csv"),
                          usecols=["KMEANS_CLUSTER"])["KMEANS_CLUSTER"].to_numpy()
    an = pd.read_csv(os.path.join(OUT_DIR, "omni_anomaly_results.csv"),
                     usecols=["ANOMALY_PRED", "ANOMALY_SCORE"])
    assert len(cluster) == len(an) == ctx.n, "outputs are not aligned with the OMNI hourly index"
    anomaly = an["ANOMALY_PRED"].to_numpy() == -1
    score = -an["ANOMALY_SCORE"].to_numpy()            # higher = more anomalous

    # 1. matching matrix
    mm = matching_matrix(cluster, ctx, fill_rows)
    print("\n    Matching matrix (k-Means k=5 vs. catalogs)")
    print(mm.to_string())

    # 2. cluster-as-detector (all clusters, all scopes) + best-on-train row for the final table
    rows = []
    for kind in ctx.kinds():
        for c in sorted(set(cluster.tolist())):
            for scope in ("train", "full", "test"):
                rows.append({"cluster": c, "target": TARGET_NAMES[kind], "scope": scope,
                             **ctx.evaluate(cluster == c, kind, scope)})
    pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT_DIR, "cluster_detector_scores.csv"),
                                       index=False)
    print("\n    Cluster chosen on the training period as the detector for each catalog class:")
    for kind in ctx.kinds():
        c = best_cluster(ctx, cluster, kind)
        print(f"      {TARGET_NAMES[kind]}: cluster {c}")
        for scope in ("full", "test"):
            record_score(f"k-Means, best cluster (#{c}, chosen on train)", kind, scope,
                         ctx.evaluate(cluster == c, kind, scope))

    # 3. cluster x anomaly cross-check (the report's claim about clusters 2 and 4)
    x = pd.DataFrame({"cluster": cluster, "flagged": anomaly, "score": an["ANOMALY_SCORE"]})
    cx = x.groupby("cluster").agg(hours=("flagged", "size"), pct_flagged=("flagged", "mean"),
                                  mean_anomaly_score=("score", "mean"), min_anomaly_score=("score", "min"))
    cx["pct_flagged"] = (100 * cx["pct_flagged"]).round(1)
    cx[["mean_anomaly_score", "min_anomaly_score"]] = cx[["mean_anomaly_score",
                                                          "min_anomaly_score"]].round(3)
    cx.to_csv(os.path.join(OUT_DIR, "cluster_vs_anomaly.csv"))
    print("\n    Cluster x Isolation Forest flag (score < 0 = outlier; the report says clusters 2 and 4 "
          f"got scores 'around -0.35' - the lowest score in the whole dataset is "
          f"{an['ANOMALY_SCORE'].min():.3f}):")
    print(cx.to_string())
    spikes = df["ABS_B"].to_numpy() > 25
    print(f"    Report's |B| > 25 nT claim: {int(spikes.sum())} hours exceed 25 nT, "
          f"{int((spikes & anomaly).sum())} of them flagged as anomalies")

    # 4. Isolation Forest at the report's setting vs. each catalog
    for kind in ctx.kinds():
        ap, auc = average_precision_score(ctx.mask[kind], score), roc_auc_score(ctx.mask[kind], score)
        print(f"    Isolation Forest score vs {TARGET_NAMES[kind]}: avg precision {ap:.3f} "
              f"(base rate {ctx.mask[kind].mean():.3f}), ROC-AUC {auc:.3f}")
        for scope in ("full", "test"):
            m = ctx.evaluate(anomaly, kind, scope)
            record_score(f"Isolation Forest (contamination={REPORT_CONTAMINATION}, as in report)",
                         kind, scope, m)

    # 5. contamination sweep (descriptive; the comparison table keeps only the report's setting)
    feature_cols = [c for c in df.columns if c not in TIME_COLS]
    X = StandardScaler().fit_transform(df[feature_cols])
    sweep, kept = [], {}
    for cont in CONTAMINATIONS:
        iso = IsolationForest(contamination=cont, random_state=42, n_jobs=-1)
        pred = iso.fit_predict(X) == -1
        kept[cont] = pred
        for kind in ctx.kinds():
            for scope in ("full", "test"):
                sweep.append({"contamination": cont, "target": TARGET_NAMES[kind], "scope": scope,
                              "hours_flagged": int(pred.sum()), **ctx.evaluate(pred, kind, scope)})
    sw = pd.DataFrame(sweep).round(4)
    sw.to_csv(os.path.join(OUT_DIR, "if_contamination_sweep.csv"), index=False)
    fig, axes = plt.subplots(1, len(ctx.kinds()), figsize=(6.5 * len(ctx.kinds()), 4.3),
                             squeeze=False)
    for ax, kind in zip(axes[0], ctx.kinds()):
        s = sw[(sw.target == TARGET_NAMES[kind]) & (sw.scope == "full")]
        for col, style in (("event_precision", "o-"), ("event_recall", "s-"), ("event_f1", "^-")):
            ax.plot(s.contamination, s[col], style, label=col.replace("_", " "))
        ax.axvline(REPORT_CONTAMINATION, color="gray", ls=":", label="report setting")
        ax.set(xlabel="contamination", ylabel="score", ylim=(0, 1),
               title=f"Isolation Forest vs. {TARGET_NAMES[kind]} (event level, 2012-2014)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "if_contamination_sweep.png"), dpi=200)
    plt.close(fig)

    # 6. strong anomaly runs that match no catalog event
    times = hour_timestamps(df)
    out = []
    for a, b in runs_from_mask(anomaly):
        if ctx.run_overlaps_catalog((a, b)):
            continue
        seg = slice(a, b + 1)
        ok = ~fill_rows[seg]
        out.append({"start": times.iloc[a], "end": times.iloc[b] + pd.Timedelta(hours=1),
                    "hours": b - a + 1, "max_anomaly_score": round(float(score[seg].max()), 4),
                    "mean_B_nT": round(float(df["ABS_B"].iloc[seg].mean()), 2),
                    "mean_speed_kms": round(float(df["FLOW_SPEED"].iloc[seg][ok].mean()), 1)
                        if ok.any() else np.nan,
                    "contains_fill_value": bool(fill_rows[seg].any())})
    unc = pd.DataFrame(out)
    if len(unc):
        unc = unc.sort_values(["contains_fill_value", "max_anomaly_score"],
                              ascending=[True, False]).head(TOP_N_UNCATALOGED)
    unc.to_csv(os.path.join(OUT_DIR, "uncataloged_anomalies.csv"), index=False)
    print("\n    Strongest anomaly runs matching no catalog event (fill-free first) - candidates "
          "for manual inspection:")
    print(unc.head(8).to_string(index=False) if len(unc) else "    (none)")

    # 7. overlay plots
    _overlay_plots(ctx, df, cluster, anomaly)
    print("\n    Saved matching_matrix_kmeans, cluster_detector_scores, cluster_vs_anomaly, "
          "if_contamination_sweep, uncataloged_anomalies and overlay_*.png")


if __name__ == "__main__":
    run_cluster_anomaly_comparison()