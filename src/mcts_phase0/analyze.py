"""CLI: load collected snapshots and produce the Phase 0 go/no-go analysis --
per-layer/per-metric table, scatter plot, and an explicit gate call.

Consumes what collect.py wrote, so the statistical slicing (same-step vs
all-pairs, z-scored vs raw, per-dataset vs pooled) can be re-run freely
without re-paying the generation cost.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: write files, never try to open a window
import matplotlib.pyplot as plt  # noqa: E402

from .geometry import (  # noqa: E402
    analyze_dataset,
    apply_zscore,
    build_pairs,
    cosine_distance,
    fit_zscore,
    gate_passes,
    l2_distance,
)


def _pair_arrays(records, layer, metric, z_scored, same_step_only):
    vectors = np.stack([r.hidden[layer] for r in records])
    if z_scored:
        mean, std = fit_zscore(vectors)
        vectors = apply_zscore(vectors, mean, std)
    dist_fn = cosine_distance if metric == "cosine" else l2_distance
    pairs = build_pairs(records, same_step_only=same_step_only)
    distances = np.array([dist_fn(vectors[i], vectors[j]) for i, j in pairs])
    abs_dv = np.array([abs(records[i].v_hat - records[j].v_hat) for i, j in pairs])
    return distances, abs_dv


def make_scatter(records_by_dataset, layer, metric, z_scored, out_path):
    datasets = sorted(records_by_dataset)
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        recs = records_by_dataset[ds]
        d, dv = _pair_arrays(recs, layer, metric, z_scored, same_step_only=True)
        if len(d) == 0:
            ax.set_title(f"{ds}: no pairs")
            continue
        ax.scatter(d, dv, alpha=0.35, s=14)
        cutoff = np.quantile(d, 0.10)
        ax.axvline(cutoff, ls="--", c="tab:red", label=f"bottom-decile cutoff={cutoff:.3f}")
        ax.axhline(0.3, ls=":", c="tab:orange", label="|dV|=0.3 false-merge line")
        ax.set_xlabel(f"{metric} distance{' (z-scored)' if z_scored else ''}")
        ax.set_ylabel("|dV-hat|")
        ax.set_title(f"{ds} -- layer {layer}, same-step pairs (n={len(d)})")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Phase 0 analysis: value-vs-distance geometry")
    ap.add_argument("--input", required=True, help="pickle written by collect.py")
    ap.add_argument("--output-dir", default="results/analysis")
    ap.add_argument("--rho-threshold", type=float, default=0.3)
    ap.add_argument("--false-merge-threshold", type=float, default=0.15)
    args = ap.parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)
    records = data["records"]
    layers = list(data["layers"].keys())

    by_dataset: dict[str, list] = {}
    for r in records:
        by_dataset.setdefault(r.dataset, []).append(r)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loaded {len(records)} records; layers={layers}")
    for ds, recs in sorted(by_dataset.items()):
        v = np.array([r.v_hat for r in recs])
        print(f"  {ds}: n={len(recs)} V-hat mean={v.mean():.3f} std={v.std():.3f} "
              f"nonzero={(v > 0).sum()}/{len(v)} unique={len(set(v.tolist()))}")

    frames = []
    for ds, recs in sorted(by_dataset.items()):
        df = analyze_dataset(recs, layers)
        df.insert(0, "dataset", ds)
        frames.append(df)

    import pandas as pd

    table = pd.concat(frames, ignore_index=True)
    table_path = out_dir / "per_layer_table.csv"
    table.to_csv(table_path, index=False)

    primary = table[table["same_step_only"] == True]  # noqa: E712
    print("\n=== PRIMARY (same-step-index pairs) ===")
    cols = ["dataset", "layer", "metric", "z_scored", "n_pairs", "spearman_rho", "false_merge_rate"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(primary[cols].to_string(index=False))

    passing = gate_passes(table, args.rho_threshold, args.false_merge_threshold)
    print(f"\n=== GATE: rho > {args.rho_threshold} AND false-merge < {args.false_merge_threshold} ===")
    if len(passing) == 0:
        print("NO configuration passes the gate.")
    else:
        print(passing[cols].to_string(index=False))

    for layer in layers:
        safe = layer.replace("/", "")
        for z in (False, True):
            suffix = "z" if z else "raw"
            make_scatter(by_dataset, layer, "cosine", z, out_dir / f"scatter_{safe}_cosine_{suffix}.png")

    print(f"\nwrote table -> {table_path}")
    print(f"wrote scatter plots -> {out_dir}")


if __name__ == "__main__":
    main()
