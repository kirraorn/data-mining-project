"""
Stage 7 - Comparison 2: decision-tree rules vs. Lepping-style thresholds.
(Paper: "Compare decision tree rules against historical thresholds (Lepping)".)

Three rule sets are compared term by term:
  A. Lepping-style thresholds (the proposal's non-ML baseline). The proposal gives no numeric cut-offs
     and the numbers below were NOT copied from Lepping et al. (2005): they are conventional
     magnetic-cloud-like values fixed in advance (never tuned on the catalog). Check them against the
     paper before citing, and describe them in the report as an approximation.
  B. The REPORT'S tree (src/p4_rules.py): depth 4, fit to the k-Means cluster ids. It is refit here
     with identical settings and checked against outputs/extracted_decision_rules.txt. Its leaves for
     the cluster that best matches each catalog class become explicit rules.
  C. A catalog-label tree (depth 3 and 4): quiet / ICME / CIR-HSS targets, trained before 2013-07-01
     and scored after it. This is the proposal's "separate ICME from CIR profiles" tree.

Also scores each rule set as an event detector so it appears in the final comparison table.

Outputs: outputs/rules_comparison.txt, outputs/catalog_decision_rules.txt,
         outputs/catalog_tree_confusion_test.csv (+ rows in outputs/method_scores.csv)
"""
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text

from src.compare_utils import CLASS_OF, OUT_DIR, TARGET_NAMES, TIME_COLS, best_cluster, record_score
from src.p5_catalogs import get_context

# ---- Lepping-style thresholds, applied to 6-hour rolling features ------------------------------
LEPPING_STYLE = {
    "ABS_B_roll_mean_6h": (">=", 10.0),        # enhanced field magnitude, nT
    "PLASMA_BETA_roll_mean_6h": ("<=", 0.5),   # low plasma beta
    "ABS_B_roll_std_6h": ("<=", 2.0),          # smooth field (small short-term variability), nT
}


def lepping_style_mask(df):
    mask = np.ones(len(df), dtype=bool)
    for col, (op, thr) in LEPPING_STYLE.items():
        mask &= (df[col] >= thr) if op == ">=" else (df[col] <= thr)
    return mask.to_numpy()


# ----------------------------------------------------------------------------- tree helpers
def fit_paper_tree():
    """Refit src/p4_rules.py's tree exactly. Returns (tree, X, y, feature_cols, same_as_saved)."""
    df = pd.read_csv(os.path.join(OUT_DIR, "omni_clustered_results.csv"))
    exclude = ["YEAR", "DOY", "HR", "KMEANS_CLUSTER", "Timestamp", "date", "time"]
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].dropna()
    y = df.loc[X.index, "KMEANS_CLUSTER"].to_numpy()
    dt = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X, y)
    saved = os.path.join(OUT_DIR, "extracted_decision_rules.txt")
    same = (os.path.exists(saved) and open(saved).read().strip()
            == export_text(dt, feature_names=feature_cols).strip())
    return dt, X, y, feature_cols, same


def fit_catalog_tree(ctx, df, feature_cols, depth):
    """Class-balanced tree on the catalog labels, trained on the training period only."""
    tr_lo, tr_hi = ctx.bounds("train")
    dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=20,
                                class_weight="balanced", random_state=42)
    dt.fit(df.loc[tr_lo:tr_hi - 1, feature_cols], ctx.label[tr_lo:tr_hi])
    return dt


def leaf_paths(tree, feature_names, target_class):
    """[(conditions, leaf_node)] for every leaf whose predicted class is `target_class`."""
    t, out = tree.tree_, []

    def rec(node, conds):
        if t.children_left[node] == -1:
            if tree.classes_[int(np.argmax(t.value[node][0]))] == target_class:
                out.append((conds, node))
            return
        name, thr = feature_names[t.feature[node]], t.threshold[node]
        rec(t.children_left[node], conds + [f"{name} <= {thr:.4g}"])
        rec(t.children_right[node], conds + [f"{name} > {thr:.4g}"])

    rec(0, [])
    return out


def _vars_used(paths):
    used = {}
    for conds, _ in paths:
        for c in conds:
            used.setdefault(c.split(" ")[0], []).append(c)
    return used


def _term_by_term(used):
    lines = []
    for col, (op, thr) in LEPPING_STYLE.items():
        base = col.split("_roll")[0]
        related = sorted({v for v in used if v.startswith(base)})
        lines.append(f"    {col} {op} {thr}  ->  tree uses: "
                     f"{', '.join(related) if related else 'nothing comparable'}")
    only = [v for v in sorted(used)
            if not any(v.startswith(c.split('_roll')[0]) for c in LEPPING_STYLE)]
    if only:
        lines.append(f"    Tree-only terms (not in the Lepping-style rule): {', '.join(only)}")
    beta = [float(m.group(1)) for c in sum((v for k, v in used.items() if k.startswith("PLASMA_BETA")), [])
            for m in [re.search(r"([-\d.e+]+)$", c)] if m]
    if beta and max(beta) > 10:
        lines.append(f"    NOTE: the tree's beta thresholds ({', '.join(f'{b:.4g}' for b in beta)}) are "
                     "far above Lepping's low-beta criterion (~0.5). They separate OMNI fill-value "
                     "rows (beta = 999.99), so they are not a physical low-beta test.")
    return lines


def _describe_leaves(paths, tree, leaf_ids, truth_cluster=None, catalog=None, label_names=("share",)):
    lines = []
    for conds, node in sorted(paths, key=lambda p: -int((leaf_ids == p[1]).sum())):
        in_leaf = leaf_ids == node
        stats = []
        if truth_cluster is not None:
            stats.append(f"{100 * truth_cluster[in_leaf].mean():.0f}% in target cluster")
        if catalog is not None:
            for name, arr in catalog.items():
                stats.append(f"{100 * arr[in_leaf].mean():.0f}% {name}")
        lines.append(f"    {' AND '.join(conds)}\n        [{int(in_leaf.sum())} h; "
                     f"{'; '.join(stats)}]")
    return lines or ["    (no leaf predicts this class)"]


# ----------------------------------------------------------------------------- main stage
def run_rule_comparison():
    print("Comparison 2: decision-tree rules vs. Lepping-style thresholds...")
    ctx = get_context()
    df = pd.read_csv(os.path.join(OUT_DIR, "processed_omni_features.csv"))
    feature_cols = [c for c in df.columns if c not in TIME_COLS]
    out = ["RULE COMPARISON: DECISION TREES VS. LEPPING-STYLE THRESHOLDS", "=" * 66, "",
           "A. Lepping-style rule (fixed in advance; approximate - see src/p7_compare_rules.py)"]
    out += [f"    {col} {op} {thr}" for col, (op, thr) in LEPPING_STYLE.items()]

    # ---- A. Lepping-style detector ----------------------------------------------------------
    flag = lepping_style_mask(df)
    for scope in ("full", "test"):
        record_score("Lepping-style thresholds (non-ML baseline)", "ICME", scope,
                     ctx.evaluate(flag, "ICME", scope))
    print(f"    Lepping-style rule flags {int(flag.sum())} hours")

    # ---- B. the report's tree ---------------------------------------------------------------
    dt, X, y, p_cols, same = fit_paper_tree()
    print(f"    Report's tree refit identical to outputs/extracted_decision_rules.txt: {same}")
    idx = X.index.to_numpy()
    cluster_all = np.full(ctx.n, -1)
    cluster_all[idx] = y
    pred_cluster = np.full(ctx.n, -1)
    pred_cluster[idx] = dt.predict(X)
    leaf_ids = dt.apply(X)
    fidelity = accuracy_score(y, pred_cluster[idx])
    root_feat, root_thr = p_cols[dt.tree_.feature[0]], dt.tree_.threshold[0]
    above = (X[root_feat] > root_thr).to_numpy()

    out += ["", "B. The report's tree (depth 4, fit to the k-Means cluster ids)",
            f"    Identical to the tree saved by src/p4_rules.py: {same}",
            f"    Reproduces the k-Means labels with {100 * fidelity:.1f}% accuracy "
            "(trained and scored on all hours, as in the report)",
            f"    Root split: {root_feat} <= {root_thr:.4g}. Report says hours above it 'connect "
            "directly towards Cluster 0':"]
    for c in sorted(set(y.tolist())):
        out.append(f"        cluster {c}: {100 * (y[above] == c).mean():5.1f}% of the "
                   f"{int(above.sum())} hours above the split")
    chosen = {kind: best_cluster(ctx, cluster_all, kind) for kind in ctx.kinds()}
    by_cluster = {}
    for kind, c in chosen.items():
        by_cluster.setdefault(c, []).append(kind)
    for c, ks in by_cluster.items():
        for kind in ks:
            for scope in ("full", "test"):
                record_score(f"Decision tree on k-Means labels (report's tree, cluster-{c} leaves)",
                             kind, scope, ctx.evaluate(pred_cluster == c, kind, scope))
        paths = leaf_paths(dt, p_cols, c)
        names = " and ".join(TARGET_NAMES[k] for k in ks)
        out += ["", f"    Leaves predicting cluster {c} (the cluster that best detects {names} "
                    "on the training period):"]
        out += _describe_leaves(paths, dt, leaf_ids, truth_cluster=(y == c),
                                catalog={f"catalog {TARGET_NAMES[k]}": ctx.mask[k][idx] for k in ks})
        out += ["    Term by term vs. Lepping-style:"] + _term_by_term(_vars_used(paths))
    out += ["    Caveat: this tree describes k-Means clusters, not catalog events, and k-Means "
            "clusters 2-4 are mostly OMNI fill values, so some leaves encode data-quality "
            "artifacts rather than physics."]

    # ---- C. catalog-label trees -------------------------------------------------------------
    te_lo, te_hi = ctx.bounds("test")
    tr_lo, tr_hi = ctx.bounds("train")
    rules_txt = []
    for depth in (3, 4):
        ct = fit_catalog_tree(ctx, df, feature_cols, depth)
        pred = np.zeros(ctx.n, dtype=int)
        pred[te_lo:te_hi] = ct.predict(df.loc[te_lo:te_hi - 1, feature_cols])
        pred_tr = ct.predict(df.loc[tr_lo:tr_hi - 1, feature_cols])
        for kind in ctx.kinds():
            m = ctx.evaluate(pred == CLASS_OF[kind], kind, "test")
            record_score(f"Decision tree depth {depth} (trained on catalog labels)", kind, "test", m)
            print(f"    Catalog tree depth {depth}, {TARGET_NAMES[kind]} (test): event "
                  f"P={m['event_precision']:.2f} R={m['event_recall']:.2f} F1={m['event_f1']:.2f}")
        if depth == 4:
            rules_txt.append(export_text(ct, feature_names=feature_cols))
            names = ["quiet", "ICME", "CIR/HSS"][:len(ct.classes_)]
            cm = confusion_matrix(ctx.label[te_lo:te_hi], pred[te_lo:te_hi],
                                  labels=list(range(len(names))))
            pd.DataFrame(cm, index=[f"true_{n}" for n in names],
                         columns=[f"pred_{n}" for n in names]).to_csv(
                os.path.join(OUT_DIR, "catalog_tree_confusion_test.csv"))
            out += ["", "C. Catalog-label tree, depth 4 (quiet / ICME / CIR-HSS; trained before "
                        f"{ctx.times[te_lo].date()})",
                    f"    Test-period 3-class accuracy: "
                    f"{100 * accuracy_score(ctx.label[te_lo:te_hi], pred[te_lo:te_hi]):.1f}% "
                    f"(train {100 * accuracy_score(ctx.label[tr_lo:tr_hi], pred_tr):.1f}%)"]
            leaf_tr = ct.apply(df.loc[tr_lo:tr_hi - 1, feature_cols])
            used_by_kind = {}
            for kind in ctx.kinds():
                paths = leaf_paths(ct, feature_cols, CLASS_OF[kind])
                used_by_kind[kind] = _vars_used(paths)
                out += ["", f"    Leaves predicting {TARGET_NAMES[kind]} (training hours):"]
                out += _describe_leaves(
                    paths, ct, leaf_tr,
                    catalog={f"truly {TARGET_NAMES[kind]}": ctx.mask[kind][tr_lo:tr_hi]})
                if kind == "ICME":
                    out += ["    Term by term vs. Lepping-style:"] + _term_by_term(used_by_kind[kind])
            if len(used_by_kind) == 2:
                only_i = sorted(set(used_by_kind["ICME"]) - set(used_by_kind["CIR"]))
                only_c = sorted(set(used_by_kind["CIR"]) - set(used_by_kind["ICME"]))
                out += ["", "    Variables that separate ICME leaves from CIR/HSS leaves:",
                        f"      ICME only   : {', '.join(only_i) or '-'}",
                        f"      CIR/HSS only: {', '.join(only_c) or '-'}"]

    with open(os.path.join(OUT_DIR, "catalog_decision_rules.txt"), "w") as f:
        f.write("\n\n".join(rules_txt))
    with open(os.path.join(OUT_DIR, "rules_comparison.txt"), "w") as f:
        f.write("\n".join(out))
    print("\n" + "\n".join(out[:34]))
    print("\n    Saved rules_comparison.txt, catalog_decision_rules.txt, "
          "catalog_tree_confusion_test.csv")


if __name__ == "__main__":
    run_rule_comparison()