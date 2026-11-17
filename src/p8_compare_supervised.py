"""
Stage 8 - Comparison 3: a supervised Random Forest vs. the unsupervised methods, plus the final
comparison tables and the merged per-hour comparison dataset.
(Paper: "Train a random forest and contrast its supervised accuracy against the unsupervised k-means
and decision tree".)

  * Random Forest (supervised ceiling): 3-class (quiet / ICME / CIR-HSS) on the same 80 features,
    trained before 2013-07-01 and scored after it (48 h gap).
  * Classification table (test period): accuracy, macro-F1 and per-class hour-level scores for the
    Random Forest, the catalog-label tree, k-Means clusters mapped to classes by enrichment, the
    report's tree mapped the same way, and an always-quiet baseline.
  * Event-level tables: every method on the same event-level precision / recall / F1.
  * outputs/comparison_dataset.csv: one row per hour with the catalog labels and every method's
    output, so any further comparison can be made without re-running the models.

Outputs: outputs/classification_comparison_test.csv/.md, final_comparison_<target>_<scope>.csv/.md/.png,
         rf_feature_importance.csv, rf_confusion_test.csv, comparison_dataset.csv
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src.compare_utils import (CLASS_OF, OUT_DIR, SCORES_PATH, TARGET_NAMES, TIME_COLS,
                               enrichment_map, fill_flag, hour_timestamps, record_score)
from src.p5_catalogs import get_context
from src.p7_compare_rules import fit_catalog_tree, fit_paper_tree, lepping_partial_masks

TABLE_COLS = ["method", "event_precision", "event_recall", "event_f1", "hour_precision",
              "hour_recall", "hour_f1", "hour_lift", "n_pred_events", "n_catalog_events"]


def _class_scores(truth, pred, kinds):
    labels = [0, 1, 2] if "CIR" in kinds else [0, 1]
    out = {"accuracy": accuracy_score(truth, pred),
           "macro_f1": f1_score(truth, pred, labels=labels, average="macro", zero_division=0)}
    p, r, f, _ = precision_recall_fscore_support(truth, pred, labels=labels, zero_division=0)
    for kind in kinds:
        i = CLASS_OF[kind]
        name = TARGET_NAMES[kind].replace("/", "_")
        out.update({f"{name}_precision": p[i], f"{name}_recall": r[i], f"{name}_f1": f[i]})
    return out


def run_supervised_comparison():
    print("Comparison 3: supervised Random Forest vs. unsupervised methods...")
    ctx = get_context()
    kinds = ctx.kinds()
    df = pd.read_csv(os.path.join(OUT_DIR, "processed_omni_features.csv"))
    feature_cols = [c for c in df.columns if c not in TIME_COLS]
    tr_lo, tr_hi = ctx.bounds("train")
    te_lo, te_hi = ctx.bounds("test")
    truth_te = ctx.label[te_lo:te_hi]

    # ---- trivial reference rows ---------------------------------------------------------------
    for kind in kinds:
        for scope in ("full", "test"):
            record_score("Trivial: flag every hour", kind, scope,
                         ctx.evaluate(np.ones(ctx.n, dtype=bool), kind, scope))

    # ---- Random Forest (3-class) --------------------------------------------------------------
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1,
                                class_weight="balanced_subsample", random_state=42)
    rf.fit(df.loc[tr_lo:tr_hi - 1, feature_cols], ctx.label[tr_lo:tr_hi])
    rf_pred = np.zeros(ctx.n, dtype=int)
    rf_pred[te_lo:te_hi] = rf.predict(df.loc[te_lo:te_hi - 1, feature_cols])
    for kind in kinds:
        m = ctx.evaluate(rf_pred == CLASS_OF[kind], kind, "test")
        record_score("Random Forest (supervised ceiling)", kind, "test", m)
        print(f"    Random Forest, {TARGET_NAMES[kind]} (test): event P={m['event_precision']:.2f} "
              f"R={m['event_recall']:.2f} F1={m['event_f1']:.2f}")
    names = ["quiet", "ICME", "CIR/HSS"][:3 if "CIR" in kinds else 2]
    pd.DataFrame(confusion_matrix(truth_te, rf_pred[te_lo:te_hi], labels=range(len(names))),
                 index=[f"true_{n}" for n in names],
                 columns=[f"pred_{n}" for n in names]).to_csv(
        os.path.join(OUT_DIR, "rf_confusion_test.csv"))
    pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15) \
        .round(4).to_csv(os.path.join(OUT_DIR, "rf_feature_importance.csv"), header=["importance"])

    # ---- other methods as 3-class predictors ---------------------------------------------------
    cluster = pd.read_csv(os.path.join(OUT_DIR, "omni_clustered_results.csv"),
                          usecols=["KMEANS_CLUSTER"])["KMEANS_CLUSTER"].to_numpy()
    km_map = enrichment_map(ctx, cluster)
    km_pred = np.array([km_map[c] for c in cluster])

    dt, X, y, p_cols, _ = fit_paper_tree()
    paper_cluster = np.full(ctx.n, -1)
    paper_cluster[X.index.to_numpy()] = dt.predict(X)
    pt_map = enrichment_map(ctx, paper_cluster)
    pt_pred = np.array([pt_map.get(c, 0) for c in paper_cluster])

    ct = fit_catalog_tree(ctx, df, feature_cols, 4)
    ct_pred = np.zeros(ctx.n, dtype=int)
    ct_pred[te_lo:te_hi] = ct.predict(df.loc[te_lo:te_hi - 1, feature_cols])

    cls = {"Random Forest (supervised)": rf_pred,
           "Decision tree depth 4 (trained on catalog labels)": ct_pred,
           "k-Means k=5 (clusters mapped to classes by enrichment)": km_pred,
           "Report's tree (k-Means labels, mapped by enrichment)": pt_pred,
           "Always 'quiet' (baseline)": np.zeros(ctx.n, dtype=int)}
    rows = [{"method": m, **_class_scores(truth_te, p[te_lo:te_hi], kinds)} for m, p in cls.items()]
    ct_tab = pd.DataFrame(rows).round(3)
    ct_tab.to_csv(os.path.join(OUT_DIR, "classification_comparison_test.csv"), index=False)
    print(f"\n    3-class comparison, test period ({te_hi - te_lo} hours; cluster->class maps: "
          f"k-Means {km_map}, report's tree {pt_map})")
    print(ct_tab.to_string(index=False))
    try:
        open(os.path.join(OUT_DIR, "classification_comparison_test.md"), "w").write(
            ct_tab.to_markdown(index=False))
    except ImportError:
        pass

    # ---- final event-level tables ---------------------------------------------------------------
    sc = pd.read_csv(SCORES_PATH)
    for target in sc.target.unique():
        for scope in ("test", "full"):
            sub = sc[(sc.target == target) & (sc.scope == scope)][TABLE_COLS]
            if sub.empty:
                continue
            sub = sub.sort_values("event_f1", ascending=False)
            tag = f"{target.replace('/', '-')}_{scope}"
            sub.to_csv(os.path.join(OUT_DIR, f"final_comparison_{tag}.csv"), index=False)
            try:
                open(os.path.join(OUT_DIR, f"final_comparison_{tag}.md"), "w").write(
                    sub.round(3).to_markdown(index=False))
            except ImportError:
                pass
            print(f"\n    {target} detection - {scope} period (event-level, sorted by event F1)")
            print(sub.round(3).to_string(index=False))
            fig, ax = plt.subplots(figsize=(9, 0.5 * len(sub) + 1.6))
            ax.barh(sub.method[::-1], sub.event_f1[::-1], color="tab:blue")
            ax.set(xlabel="event-level F1", xlim=(0, 1),
                   title=f"{target} detection, {scope} period")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, f"final_comparison_{tag}.png"), dpi=180)
            plt.close(fig)

    # ---- merged per-hour comparison dataset ----------------------------------------------------
    an = pd.read_csv(os.path.join(OUT_DIR, "omni_anomaly_results.csv"),
                     usecols=["ANOMALY_PRED", "ANOMALY_SCORE"])
    ds = df[TIME_COLS].copy()
    ds["TIMESTAMP"] = hour_timestamps(df).values
    ds["SPLIT"] = ctx.split_names()
    ds["LABEL"] = ctx.label                      # 0 quiet, 1 ICME, 2 CIR/HSS
    ds["ICME"] = ctx.mask["ICME"].astype(int)
    ds["CIR_HSS"] = ctx.mask["CIR"].astype(int)
    ds["FILL_ROW"] = fill_flag(df).astype(int)
    ds["KMEANS_CLUSTER"] = cluster
    ds["KMEANS_CLASS"] = km_pred
    ds["REPORT_TREE_CLUSTER"] = paper_cluster
    ds["IF_ANOMALY"] = (an["ANOMALY_PRED"].to_numpy() == -1).astype(int)
    ds["IF_SCORE"] = an["ANOMALY_SCORE"].to_numpy()
    lepping_masks = lepping_partial_masks(df)
    ds["LEPPING_LOOSE_PARTIAL"] = lepping_masks["Loose"].astype(int)
    ds["LEPPING_STRICT_PARTIAL"] = lepping_masks["Strict"].astype(int)
    ds["CATALOG_TREE_CLASS"] = np.where(ds["SPLIT"] == "test", ct_pred, -1)
    ds["RF_CLASS"] = np.where(ds["SPLIT"] == "test", rf_pred, -1)
    ds.to_csv(os.path.join(OUT_DIR, "comparison_dataset.csv"), index=False)
    print(f"\n    Saved comparison_dataset.csv ({len(ds)} rows x {ds.shape[1]} columns; "
          "RF / catalog-tree classes are -1 outside the test period)")


if __name__ == "__main__":
    run_supervised_comparison()
