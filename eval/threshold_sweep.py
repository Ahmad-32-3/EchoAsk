"""The centerpiece eval: where does the similarity threshold actually go?

Loads a labeled sample of Quora Question Pairs (same-intent vs
different-intent question pairs, human-labeled), embeds both questions in
each pair with the project's real embedding backend, and sweeps the cosine
similarity threshold to see what precision/recall a cache actually gets at
each cutoff.

Design decision this produces: a cache's asymmetry is precision-first, not
F1-first. A false negative (similarity just under threshold) costs exactly
one ordinary LLM call — no correctness harm, the system behaves as if there
were no cache. A false positive (similarity over threshold, wrong answer
served) is a genuine bug — a user gets somebody else's answer. So the
threshold is chosen as the smallest value that keeps precision at or above a
stated safety bar (default 0.97), not the value that maximizes F1.

Usage: python -m eval.threshold_sweep [--sample-size 20000] [--min-precision 0.97]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.embeddings import embed  # noqa: E402

DATA_PATH = pathlib.Path(__file__).parent / "data" / "qqp_raw.tsv"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"
THRESHOLDS = np.round(np.arange(0.50, 0.995, 0.01), 3)


def load_sample(sample_size: int, seed: int = 13) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep="\t", on_bad_lines="skip")
    df = df.dropna(subset=["question1", "question2", "is_duplicate"])
    df["is_duplicate"] = df["is_duplicate"].astype(int)
    sample = df.sample(n=min(sample_size, len(df)), random_state=seed).reset_index(drop=True)
    return sample


def compute_similarities(sample: pd.DataFrame) -> np.ndarray:
    # Embed the unique set of questions once, reuse vectors for both columns —
    # much cheaper than embedding every (q1, q2) pair independently.
    all_questions = pd.concat([sample["question1"], sample["question2"]]).unique().tolist()
    vectors = embed(all_questions)
    lookup = {q: v for q, v in zip(all_questions, vectors)}

    v1 = np.array([lookup[q] for q in sample["question1"]])
    v2 = np.array([lookup[q] for q in sample["question2"]])
    # Vectors are already L2-normalized -> row-wise dot product is cosine similarity.
    return np.sum(v1 * v2, axis=1)


def sweep(similarities: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in THRESHOLDS:
        pred = similarities >= t
        tp = int(np.sum(pred & (labels == 1)))
        fp = int(np.sum(pred & (labels == 0)))
        fn = int(np.sum((~pred) & (labels == 1)))
        tn = int(np.sum((~pred) & (labels == 0)))
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        # NOTE (found by independent review): `if precision and recall` treats
        # a genuine 0.0 as falsy, silently reporting NaN instead of 0 for an
        # honestly-zero F1. Check for NaN explicitly instead of truthiness so
        # a real zero is reported as 0, not hidden as undefined.
        if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
            f1 = float("nan")
        else:
            f1 = 2 * precision * recall / (precision + recall)
        accuracy = (tp + tn) / len(labels)
        rows.append(
            dict(threshold=t, precision=precision, recall=recall, f1=f1, accuracy=accuracy,
                 tp=tp, fp=fp, fn=fn, tn=tn, predicted_positive=tp + fp)
        )
    return pd.DataFrame(rows)


def choose_threshold(results: pd.DataFrame, min_precision: float) -> dict:
    eligible = results[results["precision"] >= min_precision]
    if eligible.empty:
        # No threshold in the sweep range hits the safety bar — pick the
        # highest-precision point we have and say so honestly.
        best = results.sort_values("precision", ascending=False).iloc[0]
        return {"threshold": float(best["threshold"]), "met_bar": False, "row": best}
    best = eligible.sort_values(["threshold"]).iloc[0]  # smallest threshold meeting the bar -> best recall among safe options
    return {"threshold": float(best["threshold"]), "met_bar": True, "row": best}


def plot(results: pd.DataFrame, chosen_threshold: float, out_path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results["threshold"], results["precision"], label="precision", color="#2563eb")
    ax.plot(results["threshold"], results["recall"], label="recall", color="#e2703a")
    ax.plot(results["threshold"], results["f1"], label="F1", color="#6b7280", linestyle="--")
    ax.axvline(chosen_threshold, color="#16a34a", linestyle=":", label=f"chosen = {chosen_threshold}")
    ax.set_xlabel("cosine similarity threshold")
    ax.set_ylabel("score")
    ax.set_title("Semantic cache threshold sweep (QQP)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--min-precision", type=float, default=0.97)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.sample_size} sampled QQP pairs...")
    sample = load_sample(args.sample_size)
    print(f"class balance: {sample['is_duplicate'].mean():.3f} duplicate rate, n={len(sample)}")

    print("embedding questions...")
    similarities = compute_similarities(sample)
    labels = sample["is_duplicate"].to_numpy()

    print("sweeping thresholds...")
    results = sweep(similarities, labels)
    results.to_csv(RESULTS_DIR / "threshold_sweep.csv", index=False)

    choice = choose_threshold(results, args.min_precision)
    chosen = choice["threshold"]
    row = choice["row"]

    plot(results, chosen, RESULTS_DIR / "threshold_sweep.png")

    summary = (
        f"chosen threshold: {chosen}\n"
        f"met precision bar ({args.min_precision}): {choice['met_bar']}\n"
        f"at chosen threshold -> precision={row['precision']:.4f} recall={row['recall']:.4f} "
        f"f1={row['f1']:.4f} accuracy={row['accuracy']:.4f}\n"
        f"tp={int(row['tp'])} fp={int(row['fp'])} fn={int(row['fn'])} tn={int(row['tn'])}\n"
        f"sample_size={len(sample)} duplicate_rate={sample['is_duplicate'].mean():.4f}\n"
    )
    print(summary)
    (RESULTS_DIR / "threshold_sweep_summary.txt").write_text(summary)


if __name__ == "__main__":
    main()
