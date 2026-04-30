#!/usr/bin/env python3
"""
Plot reliability diagrams (top-label and per-class) from a calibration JSON.

Default input: `calibration_logs/forget01-mcqa--phi.json`
Saves plots to: `calibration_logs/plots/` as PNGs.

Usage:
    python scripts/plot_reliability.py \
        --input calibration_logs/forget01-mcqa--phi.json \
        --bins 10

Outputs printed: ECE and MCE for top-label calibration.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
# scipy
from scipy.special import softmax



LABELS = ["A", "B", "C", "D"]


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def label_to_index(lbl):
    lbl = lbl.strip().upper()
    return ord(lbl) - ord("A")


def compute_top_label_confidence_and_correctness(entries):
    confidences = []
    correct = []
    probs_matrix = []
    true_idx = []

    for e in entries:
        # Some logs have option_probs as nested list-of-list; flatten safely
        probs = e.get("option_probs")
        if probs is None:
            logits = e.get("option_logits")
            # convert logits to probs using softmax
            probs = softmax(logits)
        if isinstance(probs, list) and len(probs) > 0 and isinstance(probs[0], list):
            probs = probs[0]
        probs = np.array(probs, dtype=float)
        probs_matrix.append(probs)

        pred_lbl = e.get("predicted_label")
        true_lbl = e.get("label")
        if pred_lbl is None or true_lbl is None:
            # fallback: assume argmax on probs corresponds to predicted
            pred_idx = int(np.argmax(probs))
            pred_lbl = LABELS[pred_idx]
            true_lbl = LABELS[int(np.argmax(probs))]
        pred_idx = label_to_index(pred_lbl)
        true_i = label_to_index(true_lbl)
        true_idx.append(true_i)

        confidences.append(float(probs[pred_idx]))
        correct.append(int(pred_idx == true_i))

    return np.array(confidences), np.array(correct), np.vstack(probs_matrix), np.array(true_idx)


def compute_binned_stats(confidences, correctness, bins=10):
    # bins: number of equal-width bins between 0 and 1
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2.0

    accuracies = np.zeros(bins, dtype=float)
    avg_confidences = np.zeros(bins, dtype=float)
    counts = np.zeros(bins, dtype=int)

    N = len(confidences)
    for i in range(bins):
        # include left edge, exclude right (except last bin)
        if i < bins - 1:
            mask = (confidences >= bin_lowers[i]) & (confidences < bin_uppers[i])
        else:
            mask = (confidences >= bin_lowers[i]) & (confidences <= bin_uppers[i])
        counts[i] = int(mask.sum())
        if counts[i] > 0:
            avg_confidences[i] = confidences[mask].mean()
            accuracies[i] = correctness[mask].mean()
        else:
            avg_confidences[i] = (bin_lowers[i] + bin_uppers[i]) / 2.0
            accuracies[i] = 0.0

    # ECE: weighted average |acc - conf|
    if N > 0:
        ece = np.sum((counts / N) * np.abs(accuracies - avg_confidences))
        mce = np.max(np.abs(accuracies - avg_confidences))
    else:
        ece = 0.0
        mce = 0.0

    return {
        "bin_centers": bin_centers,
        "accuracies": accuracies,
        "avg_confidences": avg_confidences,
        "counts": counts,
        "ece": float(ece),
        "mce": float(mce),
        "N": N,
    }


def plot_reliability(bin_centers, accuracies, avg_confidences, counts, outpath, title=None):
    width = 1.0 / len(bin_centers)
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.bar(bin_centers, accuracies, width=width * 0.95, alpha=0.8, label="Accuracy", edgecolor="k")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    if title:
        ax.set_title(title)
    ax.legend()

    # annotate counts under bars
    for x, c in zip(bin_centers, counts):
        if c > 0:
            ax.text(x, -0.06, str(int(c)), ha="center", va="top", fontsize=8, rotation=0)
    ax.set_ylim(-0.08, 1.02)
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_per_class(probs_matrix, true_idx, bins, outpath, labels=None):
    K = probs_matrix.shape[1]
    labels = labels or [chr(ord("A") + i) for i in range(K)]
    ncols = 2
    nrows = (K + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for c in range(K):
        class_probs = probs_matrix[:, c]
        class_true = (true_idx == c).astype(int)
        stats = compute_binned_stats(class_probs, class_true, bins=bins)
        ax = axes[c]
        width = 1.0 / len(stats["bin_centers"])
        ax.bar(stats["bin_centers"], stats["accuracies"], width=width * 0.95, alpha=0.8, edgecolor="k")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability for class {0}".format(labels[c]))
        ax.set_ylabel("Empirical frequency")
        ax.set_title(f"Class {labels[c]} (ECE={stats['ece']:.3f})")

    # hide any extra subplots
    for i in range(K, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="calibration_logs/forget01-mcqa--phi.json")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--outdir", type=str, default="")
    args = parser.parse_args()

    if args.outdir and not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    else:
        # args.outdir is empty, so save in same directory as input
        args.outdir = os.path.dirname(args.input)
        args.outdir = os.path.join(args.outdir, "plots")
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)



    entries = load_json(args.input)
    confidences, correct, probs_matrix, true_idx = compute_top_label_confidence_and_correctness(entries)

    top_stats = compute_binned_stats(confidences, correct, bins=args.bins)

    top_out = os.path.join(args.outdir, "reliability_top.png")
    title = f"Top-label reliability (ECE={top_stats['ece']:.4f}, MCE={top_stats['mce']:.4f}, N={top_stats['N']})"
    plot_reliability(top_stats["bin_centers"], top_stats["accuracies"], top_stats["avg_confidences"], top_stats["counts"], top_out, title=title)

    per_class_out = os.path.join(args.outdir, "reliability_per_class.png")
    plot_per_class(probs_matrix, true_idx, bins=args.bins, outpath=per_class_out, labels=LABELS[:probs_matrix.shape[1]])

    print("Saved:")
    print(" - Top-label reliability:", top_out)
    print(" - Per-class reliability:", per_class_out)
    print(f"Top-label ECE={top_stats['ece']:.6f}, MCE={top_stats['mce']:.6f}")


if __name__ == "__main__":
    main()

# USAGE:
"""
python plot_reliability.py --input saves/eval/relu_phi-1_5_retain90_batch1_debug/retain-mcqa.json --bins 10
python plot_reliability.py --input saves/eval/relu_phi-1_5_retain90_batch1_debug/forget-mcqa.json --bins 10
"""