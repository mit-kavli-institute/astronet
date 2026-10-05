#!/usr/bin/env python
"""Compare the 4-class and P-vs-notP vetting arms.

Computes planet-class metrics on:
  1. the in-distribution dec2025 test split (from the evaluation/*.npy files
     that train.py saves inside each model dir), and
  2. sector 93, scored fresh from the sector-93-scatter tfrecords and labelled
     with the Aug-5 vetting sheets in sector93_results/data/full_results/.

Threshold policy: each arm gets its own operating threshold, chosen to
maximize F1 on the *val* split, then applied frozen to test and S93.
A shared threshold (e.g. 0.215) would compare calibration, not
discrimination, because softmax and sigmoid scores live on different scales.
Threshold-free metrics (AP, ROC-AUC) are the headline numbers.

Usage:
  python evaluate_comparison.py RUN_ROOT [--out_dir DIR] [--skip_s93] ...
  (RUN_ROOT is the OUT_ROOT of run_training.sh, containing 4class/ and binary/)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

S93_TFRECORDS_DEFAULT = "/pdo/astronet-data/data/tfrecords/sector-93-scatter/*"
S93_LABELS_DIR_DEFAULT = "/pdo/users/pablomer/sector93_results/data/full_results"
S93_LABEL_SHEETS = {
    "planet": "qlp-s93-ffi-group-1vet-Aug5.csv",
    "eb": "qlp-s93-ffi-eb-1vet-Aug5.csv",
    "junk": "qlp-s93-ffi-junk-1vet-Aug5.csv",
    "ivfail": "qlp-s93-ffi-iv-fail.csv",
}

ARM_COLORS = {"4class": "#2a78d6", "binary": "#eb6834"}  # colorblind-safe pair


# ---------------------------------------------------------------------------
# Model dir discovery and score extraction
# ---------------------------------------------------------------------------

def find_member_dirs(arm_dir):
    """Model checkpoint dirs (config.json + train_flags.json) under arm_dir."""
    members = []
    for entry in sorted(os.listdir(arm_dir)):
        path = os.path.join(arm_dir, entry)
        if (os.path.isdir(path)
                and os.path.isfile(os.path.join(path, "config.json"))
                and os.path.isfile(os.path.join(path, "train_flags.json"))):
            members.append(path)
    if not members:
        raise ValueError(f"No model checkpoint dirs found under {arm_dir}")
    return members


def planet_score_and_label(pred, label):
    """Planet-class score and binary label from either arm's arrays.

    4-class: pred (N,4) softmax, label (N,4) one-hot -> column 0.
    binary:  pred (N,1) sigmoid,  label (N,)  0/1    -> ravel.
    """
    pred = np.asarray(pred)
    label = np.asarray(label)
    score = pred[:, 0] if (pred.ndim == 2 and pred.shape[1] > 1) else pred.ravel()
    y = label[:, 0] if (label.ndim == 2 and label.shape[1] > 1) else label.ravel()
    return score.astype(np.float64), y.astype(np.int64)


def load_split_scores(members, split):
    """Planet scores + labels for one split.

    Returns (mean_score, y, member_scores) where member_scores is a list of
    per-member score arrays (one per seed) and mean_score their average.
    """
    scores, y_ref = [], None
    for m in members:
        pred = np.load(os.path.join(m, "evaluation", f"{split}_pred.npy"))
        label = np.load(os.path.join(m, "evaluation", f"{split}_label.npy"))
        score, y = planet_score_and_label(pred, label)
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y, y_ref):
            raise ValueError(f"Label mismatch across members for split '{split}'")
        scores.append(score)
    return np.mean(scores, axis=0), y_ref, scores


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def f1_optimal_threshold(y, score):
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y, score)
    # precision/recall have len(thresholds)+1; drop the final (P=1, R=0) point.
    p, r = precision[:-1], recall[:-1]
    denom = np.where((p + r) > 0, p + r, 1.0)
    f1 = np.where((p + r) > 0, 2 * p * r / denom, 0.0)
    i = int(np.argmax(f1))
    return float(thresholds[i]), float(f1[i])


def metrics_at_threshold(y, score, threshold):
    from sklearn import metrics as skm
    yb = (score >= threshold).astype(int)
    tp = int(np.sum((yb == 1) & (y == 1)))
    fp = int(np.sum((yb == 1) & (y == 0)))
    fn = int(np.sum((yb == 0) & (y == 1)))
    return {
        "threshold": float(threshold),
        "precision": float(skm.precision_score(y, yb, zero_division=0)),
        "recall": float(skm.recall_score(y, yb, zero_division=0)),
        "f1": float(skm.f1_score(y, yb, zero_division=0)),
        "n_flagged": int(tp + fp),
        "tp": tp, "fp": fp, "fn": fn,
    }


def threshold_free_metrics(y, score):
    from sklearn import metrics as skm
    return {
        "n": int(len(y)),
        "n_pos": int(np.sum(y)),
        "average_precision": float(skm.average_precision_score(y, score)),
        "roc_auc": float(skm.roc_auc_score(y, score)),
    }


def precision_at_recall(y, score, recall_targets):
    """Best precision achievable with recall >= target (from the PR curve)."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y, score)
    out = {}
    for rt in recall_targets:
        ok = recall >= rt
        out[rt] = float(np.max(precision[ok])) if np.any(ok) else float("nan")
    return out


def recall_at_precision(y, score, precision_targets):
    """Best recall achievable with precision >= target (from the PR curve)."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y, score)
    out = {}
    for pt in precision_targets:
        ok = precision >= pt
        out[pt] = float(np.max(recall[ok])) if np.any(ok) else float("nan")
    return out


# ---------------------------------------------------------------------------
# Sector 93
# ---------------------------------------------------------------------------

def run_s93_inference(members, arm, tfrecord_glob, out_dir, shard_limit=0,
                      overwrite=False):
    """Score the S93 tfrecords with each member.

    Returns (mean_df, member_frames): the across-member mean per astro_id and
    the per-member frames (one per seed). Per-member results are cached as
    CSVs, so re-runs only score members that haven't been scored yet.
    """
    import tensorflow as tf
    from astronet.astro_cnn_model import input_ds
    from downstream_tasks.load_model import load_model_from_checkpoint

    preds_dir = os.path.join(out_dir, "s93_preds")
    os.makedirs(preds_dir, exist_ok=True)
    mean_csv = os.path.join(preds_dir, f"{arm}_mean.csv")

    files = sorted(tf.io.gfile.glob(tfrecord_glob))
    if shard_limit:
        files = files[:shard_limit]
    if not files:
        raise ValueError(f"No tfrecords match {tfrecord_glob}")
    print(f"[s93] {arm}: scoring {len(files)} shards with {len(members)} member(s)")

    member_frames = []
    for i, member in enumerate(members, 1):
        cache = os.path.join(
            preds_dir, f"{arm}_{os.path.basename(member)}.csv")
        if os.path.exists(cache) and not overwrite:
            print(f"[s93]   member {i}: cached ({cache})")
            member_frames.append(pd.read_csv(cache))
            continue
        model, config = load_model_from_checkpoint(member, compile_model=False)
        ds = input_ds.build_eval_dataset(
            file_pattern=files,
            input_config=config.inputs,
            batch_size=config.hparams.batch_size,
            include_identifiers=True,
            include_labels=False)
        ids, scores = [], []
        for features, astro_id in ds:
            pred = model(features, training=False).numpy()
            scores.append(pred[:, 0])
            ids.append(np.asarray(astro_id).reshape(-1))
        df = pd.DataFrame({
            "astro_id": np.concatenate(ids).astype(np.int64),
            "planet_score": np.concatenate(scores),
        })
        df.to_csv(cache, index=False)
        print(f"[s93]   member {i}/{len(members)}: {len(df)} records -> {cache}")
        member_frames.append(df)

    # Collapse duplicate astro_ids inside the S93 tfrecords themselves
    # (5962 records, 5956 unique) before averaging across members.
    member_frames = [f.groupby("astro_id", as_index=False)["planet_score"].mean()
                     for f in member_frames]
    merged = pd.concat(member_frames)
    mean_df = (merged.groupby("astro_id", as_index=False)["planet_score"]
               .mean())
    mean_df.to_csv(mean_csv, index=False)
    print(f"[s93] {arm}: {len(mean_df)} unique astro_ids -> {mean_csv}")
    return mean_df, member_frames


def load_s93_label_sets(labels_dir):
    """uid (= TIC*100 + planetno = astro_id) sets per vetting sheet."""
    sets = {}
    for name, fname in S93_LABEL_SHEETS.items():
        df = pd.read_csv(os.path.join(labels_dir, fname), skiprows=2)
        uid = (df["TIC"].astype("int64") * 100
               + df["Planet Number"].astype("int64"))
        sets[name] = set(uid.tolist())
    return sets


def s93_label_frames(label_sets):
    """Three labelling policies -> DataFrame(astro_id, y) each.

    unanimous     : TCEs in exactly one sheet; y=1 iff that sheet is 'planet'.
    any_planet    : all labelled TCEs; y=1 iff on the planet sheet at all.
    strict_planet : all labelled TCEs; y=1 iff ONLY on the planet sheet
                    (any competing EB/junk/iv-fail vote makes it a negative).
    """
    all_uids = sorted(set().union(*label_sets.values()))
    rows = []
    for u in all_uids:
        in_sheets = {k for k, s in label_sets.items() if u in s}
        rows.append({
            "astro_id": u,
            "n_sheets": len(in_sheets),
            "in_planet": "planet" in in_sheets,
            "planet_only": in_sheets == {"planet"},
        })
    df = pd.DataFrame(rows)
    return {
        "unanimous": (df[df.n_sheets == 1]
                      .assign(y=lambda d: d.in_planet.astype(int))
                      [["astro_id", "y"]]),
        "any_planet": df.assign(y=lambda d: d.in_planet.astype(int))
                        [["astro_id", "y"]],
        "strict_planet": df.assign(y=lambda d: d.planet_only.astype(int))
                           [["astro_id", "y"]],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def plot_pr_curves(arm_data, title, out_png, operating_points=None,
                   xlim=(0, 1.02), ylim=(0, 1.02)):
    """PR curves for both arms on one dataset. arm_data: {arm: (y, score)}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for arm, (y, score) in arm_data.items():
        precision, recall, _ = precision_recall_curve(y, score)
        ap = average_precision_score(y, score)
        ax.plot(recall, precision, lw=2, color=ARM_COLORS[arm],
                label=f"{arm} (AP={ap:.3f})")
        if operating_points and arm in operating_points:
            thr = operating_points[arm]
            m = metrics_at_threshold(np.asarray(y), np.asarray(score), thr)
            ax.plot(m["recall"], m["precision"], "o", ms=8,
                    color=ARM_COLORS[arm], mec="white", mew=1.5)
            ax.annotate(f"thr={thr:.3f}", (m["recall"], m["precision"]),
                        textcoords="offset points", xytext=(6, -12),
                        fontsize=8, color=ARM_COLORS[arm])
    baseline = np.mean(next(iter(arm_data.values()))[0])
    if ylim[0] <= baseline <= ylim[1]:
        ax.axhline(baseline, color="0.6", lw=1, ls="--",
                   label=f"prevalence ({baseline:.2f})")
    ax.set_xlabel("Recall (planet class)")
    ax.set_ylabel("Precision (planet class)")
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_png}")


def plot_pr_band(panels, out_png):
    """PR curves with an across-seed mean line and ±1 std shaded band.

    panels: list of (title, arm_pairs, xlim, ylim) where arm_pairs is
    {arm: [(y, score), ...]} — one (labels, scores) pair per seed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    grid = np.linspace(0, 1, 501)
    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 5),
                             squeeze=False)
    for ax, (title, arm_pairs, xlim, ylim) in zip(axes[0], panels):
        for arm, pairs in arm_pairs.items():
            curves, aps = [], []
            for y, score in pairs:
                precision, recall, _ = precision_recall_curve(y, score)
                # recall is decreasing; flip both for interpolation.
                curves.append(np.interp(grid, recall[::-1], precision[::-1]))
                aps.append(average_precision_score(y, score))
            curves = np.asarray(curves)
            mean_c, std_c = curves.mean(axis=0), curves.std(axis=0)
            label = f"{arm} (AP {np.mean(aps):.3f}"
            if len(pairs) > 1:
                label += f" ± {np.std(aps):.3f}, {len(pairs)} seeds"
            label += ")"
            ax.plot(grid, mean_c, lw=2, color=ARM_COLORS[arm], label=label,
                    zorder=3)
            if len(pairs) > 1:
                ax.fill_between(grid,
                                np.clip(mean_c - std_c, 0, 1),
                                np.clip(mean_c + std_c, 0, 1),
                                color=ARM_COLORS[arm], alpha=0.22, lw=0,
                                zorder=2)
        prevalence = np.mean(next(iter(arm_pairs.values()))[0][0])
        if ylim[0] <= prevalence <= ylim[1]:
            ax.axhline(prevalence, color="0.6", lw=1, ls="--",
                       label=f"prevalence ({prevalence:.2f})")
        ax.set_xlabel("Recall (planet class)")
        ax.set_ylabel("Precision (planet class)")
        ax.set_title(title)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_png}")


def plot_seed_spread(panel_data, out_png):
    """Per-seed AP dot plot, one panel per dataset.

    panel_data: list of (panel_title, {arm: [ap_per_member, ...]}).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panel_data), figsize=(4.2 * len(panel_data), 4.5),
                             squeeze=False)
    for ax, (title, arm_aps) in zip(axes[0], panel_data):
        for pos, (arm, aps) in enumerate(arm_aps.items()):
            aps = np.asarray(aps, dtype=float)
            # deterministic horizontal spread, one dot per seed
            xs = pos + (np.arange(len(aps)) - (len(aps) - 1) / 2) * 0.07
            ax.plot(xs, aps, "o", ms=8, color=ARM_COLORS[arm], mec="white",
                    mew=1.0, zorder=3)
            ax.hlines(aps.mean(), pos - 0.22, pos + 0.22,
                      color=ARM_COLORS[arm], lw=2, zorder=2)
            label = f"mean {aps.mean():.4f}"
            if len(aps) > 1:
                label += f"\n±{aps.std():.4f}"
            ax.annotate(label, (pos + 0.26, aps.mean()), fontsize=8,
                        va="center", color=ARM_COLORS[arm])
        ax.set_xticks(range(len(arm_aps)))
        ax.set_xticklabels(list(arm_aps))
        ax.set_xlim(-0.5, len(arm_aps) - 0.5 + 0.45)
        ax.set_ylabel("Planet-class AP")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_png}")


def fmt_table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_root", nargs="?", default=None,
                    help="Training OUT_ROOT containing 4class/ and binary/")
    ap.add_argument("--four_class_dir", default=None)
    ap.add_argument("--binary_dir", default=None)
    ap.add_argument("--out_dir", default=None,
                    help="Default: RUN_ROOT/comparison")
    ap.add_argument("--s93_tfrecords", default=S93_TFRECORDS_DEFAULT)
    ap.add_argument("--s93_labels_dir", default=S93_LABELS_DIR_DEFAULT)
    ap.add_argument("--s93_shard_limit", type=int, default=0,
                    help="Score only the first N shards (0 = all; smoke tests)")
    ap.add_argument("--skip_s93", action="store_true")
    ap.add_argument("--overwrite_preds", action="store_true")
    args = ap.parse_args()

    arm_dirs = {}
    if args.run_root:
        arm_dirs["4class"] = os.path.join(args.run_root, "4class")
        arm_dirs["binary"] = os.path.join(args.run_root, "binary")
    if args.four_class_dir:
        arm_dirs["4class"] = args.four_class_dir
    if args.binary_dir:
        arm_dirs["binary"] = args.binary_dir
    if set(arm_dirs) != {"4class", "binary"}:
        ap.error("Provide RUN_ROOT or both --four_class_dir and --binary_dir")

    out_dir = args.out_dir or os.path.join(
        args.run_root or os.path.dirname(arm_dirs["4class"]), "comparison")
    os.makedirs(out_dir, exist_ok=True)

    members = {arm: find_member_dirs(d) for arm, d in arm_dirs.items()}
    for arm, m in members.items():
        print(f"[setup] {arm}: {len(m)} member(s) under {arm_dirs[arm]}")

    results = {"arm_dirs": arm_dirs,
               "members": {a: [os.path.basename(x) for x in m]
                           for a, m in members.items()}}
    report = ["# 4-class vs P-vs-notP — comparison results", ""]
    report.append(f"Arms: `{arm_dirs['4class']}` vs `{arm_dirs['binary']}` "
                  f"({len(members['4class'])}/{len(members['binary'])} members)")
    report.append("")

    # ---- Threshold selection on val ------------------------------------
    thresholds = {}
    val_scores = {}
    for arm in ("4class", "binary"):
        score, y, _ = load_split_scores(members[arm], "val")
        thr, f1 = f1_optimal_threshold(y, score)
        thresholds[arm] = thr
        val_scores[arm] = (y, score)
        print(f"[val] {arm}: F1-optimal threshold {thr:.4f} (val F1 {f1:.3f})")
    results["val_thresholds"] = thresholds

    # ---- In-distribution test split ------------------------------------
    recall_targets = [0.6, 0.7, 0.8, 0.9]
    precision_targets = [0.6, 0.7, 0.8, 0.9]
    test_scores = {}
    test_rows, pr_rows, rp_rows, member_rows = [], [], [], []
    test_member_aps = {}
    test_member_pairs = {}
    results["test"] = {}
    for arm in ("4class", "binary"):
        score, y, member_scores = load_split_scores(members[arm], "test")
        test_scores[arm] = (y, score)
        test_member_pairs[arm] = [(y, ms) for ms in member_scores]
        # Per-member (seed) metrics — the spread is the yardstick for
        # whether any between-arm gap is real.
        from sklearn import metrics as skm
        m_aps, m_rocs = [], []
        for name, ms in zip(results["members"][arm], member_scores):
            m_ap = float(skm.average_precision_score(y, ms))
            m_roc = float(skm.roc_auc_score(y, ms))
            m_aps.append(m_ap)
            m_rocs.append(m_roc)
            member_rows.append([arm, name, f"{m_ap:.4f}", f"{m_roc:.4f}"])
        test_member_aps[arm] = m_aps
        if len(m_aps) > 1:
            member_rows.append([
                f"**{arm} mean ± std**", f"{len(m_aps)} seeds",
                f"{np.mean(m_aps):.4f} ± {np.std(m_aps):.4f}",
                f"{np.mean(m_rocs):.4f} ± {np.std(m_rocs):.4f}"])
        tf_m = threshold_free_metrics(y, score)
        thr_m = metrics_at_threshold(y, score, thresholds[arm])
        results["test"][arm] = {**tf_m,
                                "member_ap": m_aps, "member_roc_auc": m_rocs,
                                **{f"at_val_thr_{k}": v
                                   for k, v in thr_m.items()}}
        test_rows.append([arm, tf_m["n"], tf_m["n_pos"],
                          f"{tf_m['average_precision']:.4f}",
                          f"{tf_m['roc_auc']:.4f}",
                          f"{thr_m['threshold']:.3f}",
                          f"{thr_m['precision']:.3f}",
                          f"{thr_m['recall']:.3f}",
                          f"{thr_m['f1']:.3f}"])
        pr_rows.append([arm] + [f"{v:.3f}" for v in
                                precision_at_recall(y, score, recall_targets).values()])
        rp_rows.append([arm] + [f"{v:.3f}" for v in
                                recall_at_precision(y, score, precision_targets).values()])
    # 4-class at the production threshold, for reference.
    y4, s4 = test_scores["4class"]
    prod_m = metrics_at_threshold(y4, s4, 0.215)
    results["test"]["4class_at_production_0.215"] = prod_m
    test_rows.append(["4class @0.215 (prod)", len(y4), int(np.sum(y4)),
                      "-", "-", "0.215",
                      f"{prod_m['precision']:.3f}",
                      f"{prod_m['recall']:.3f}",
                      f"{prod_m['f1']:.3f}"])

    report += ["## In-distribution test split (dec2025_cad_scat_v5_aug)", "",
               "Thresholds are each arm's F1-optimum on the **val** split, "
               "frozen before touching test/S93. AP/ROC and @thr metrics use "
               "the across-seed ensemble-mean score.", "",
               fmt_table(test_rows, ["arm", "N", "N_pos", "AP", "ROC-AUC",
                                     "thr", "P@thr", "R@thr", "F1@thr"]),
               "",
               "**Per-seed (member) metrics:**", "",
               fmt_table(member_rows, ["arm", "member", "AP", "ROC-AUC"]),
               "",
               "**Precision at matched recall** (best P with R ≥ target):", "",
               fmt_table(pr_rows, ["arm"] + [f"P@R≥{r}" for r in recall_targets]),
               "",
               "**Recall at matched precision** (best R with P ≥ target):", "",
               fmt_table(rp_rows, ["arm"] + [f"R@P≥{p}" for p in precision_targets]),
               ""]

    plot_pr_curves(test_scores, "Planet-class PR — dec2025 test split",
                   os.path.join(out_dir, "pr_test.png"),
                   operating_points=thresholds)
    plot_pr_curves(test_scores,
                   "Planet-class PR — dec2025 test split (zoom)",
                   os.path.join(out_dir, "pr_test_zoom.png"),
                   operating_points=thresholds,
                   xlim=(0.85, 1.002), ylim=(0.85, 1.002))

    # ---- Sector 93 ------------------------------------------------------
    if not args.skip_s93:
        label_sets = load_s93_label_sets(args.s93_labels_dir)
        policies = s93_label_frames(label_sets)
        s93_runs = {arm: run_s93_inference(
                        members[arm], arm, args.s93_tfrecords, out_dir,
                        shard_limit=args.s93_shard_limit,
                        overwrite=args.overwrite_preds)
                    for arm in ("4class", "binary")}
        preds = {arm: mean_df for arm, (mean_df, _) in s93_runs.items()}

        results["s93"] = {}
        report.append("## Sector 93 (Aug-5 vetting sheets)")
        report.append("")
        report.append(
            "Scores from sector-93-scatter tfrecords; labels joined on "
            "astro_id (= TIC*100 + planetno). Caveat: the labelled set is "
            "the union of astronet top-322 and operator picks — a "
            "preselected slice, so these are comparative numbers between "
            "arms, not sector-wide absolutes.")
        report.append("")
        s93_plot_data = None
        for policy, frame in policies.items():
            rows = []
            results["s93"][policy] = {}
            plot_data = {}
            for arm in ("4class", "binary"):
                joined = frame.merge(preds[arm], on="astro_id", how="inner")
                n_missing = len(frame) - len(joined)
                y = joined["y"].to_numpy()
                score = joined["planet_score"].to_numpy()
                tf_m = threshold_free_metrics(y, score)
                thr_m = metrics_at_threshold(y, score, thresholds[arm])
                results["s93"][policy][arm] = {
                    **tf_m, "n_unscored": int(n_missing),
                    **{f"at_val_thr_{k}": v for k, v in thr_m.items()}}
                rows.append([arm, tf_m["n"], tf_m["n_pos"], n_missing,
                             f"{tf_m['average_precision']:.4f}",
                             f"{tf_m['roc_auc']:.4f}",
                             f"{thr_m['threshold']:.3f}",
                             f"{thr_m['precision']:.3f}",
                             f"{thr_m['recall']:.3f}",
                             f"{thr_m['f1']:.3f}"])
                plot_data[arm] = (y, score)
            report += [f"### Policy: {policy}", "",
                       fmt_table(rows, ["arm", "N", "N_pos", "unscored", "AP",
                                        "ROC-AUC", "thr", "P@thr", "R@thr",
                                        "F1@thr"]),
                       ""]
            if policy == "unanimous":
                s93_plot_data = plot_data
        if s93_plot_data:
            plot_pr_curves(s93_plot_data,
                           "Planet-class PR — sector 93 (unanimous labels)",
                           os.path.join(out_dir, "pr_s93_unanimous.png"),
                           operating_points=thresholds)

        # Per-member (seed) AP on the unanimous policy.
        from sklearn import metrics as skm
        s93_member_aps = {}
        s93_member_pairs = {}
        s93_member_rows = []
        unan = policies["unanimous"]
        for arm in ("4class", "binary"):
            aps = []
            s93_member_pairs[arm] = []
            for name, frame in zip(results["members"][arm],
                                   s93_runs[arm][1]):
                joined = unan.merge(frame, on="astro_id", how="inner")
                s93_member_pairs[arm].append(
                    (joined["y"].to_numpy(), joined["planet_score"].to_numpy()))
                ap = float(skm.average_precision_score(
                    joined["y"], joined["planet_score"]))
                aps.append(ap)
                s93_member_rows.append([arm, name, f"{ap:.4f}"])
            s93_member_aps[arm] = aps
            if len(aps) > 1:
                s93_member_rows.append([
                    f"**{arm} mean ± std**", f"{len(aps)} seeds",
                    f"{np.mean(aps):.4f} ± {np.std(aps):.4f}"])
            results["s93"]["unanimous"][arm]["member_ap"] = aps
        report += ["**Per-seed (member) AP, unanimous policy:**", "",
                   fmt_table(s93_member_rows, ["arm", "member", "AP"]), ""]

        plot_seed_spread(
            [("dec2025 test split", test_member_aps),
             ("sector 93 (unanimous)", s93_member_aps)],
            os.path.join(out_dir, "ap_seed_spread.png"))
        plot_pr_band(
            [("dec2025 test split (zoom)", test_member_pairs,
              (0.85, 1.002), (0.85, 1.002)),
             ("sector 93 (unanimous)", s93_member_pairs,
              (0, 1.02), (0, 1.02))],
            os.path.join(out_dir, "pr_band.png"))
    else:
        plot_seed_spread([("dec2025 test split", test_member_aps)],
                         os.path.join(out_dir, "ap_seed_spread.png"))
        plot_pr_band(
            [("dec2025 test split (zoom)", test_member_pairs,
              (0.85, 1.002), (0.85, 1.002))],
            os.path.join(out_dir, "pr_band.png"))

    # ---- Save ----------------------------------------------------------
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"\n[done] Results in {out_dir}/ (results.md, results.json, *.png)")


if __name__ == "__main__":
    main()
