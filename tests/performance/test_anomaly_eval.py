"""Evaluation harness for the anomaly detector against synthetic labeled data.

The original implementation used a hardcoded contamination=0.1 with no
evidence it actually catches injected outliers at a usable precision/recall.
This builds a labeled synthetic dataset (known "normal" transaction amounts
plus known-injected outliers), runs detect_anomalies() against it at several
contamination settings, and asserts a minimum precision/recall bar so a
future change to the detector logic can't silently regress it.

Run directly for a human-readable report:
    python tests/performance/test_anomaly_eval.py

Or as part of the normal test suite:
    pytest tests/performance/test_anomaly_eval.py -v
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from financial_doc_tool.core.anomaly import detect_anomalies


@dataclass
class EvalResult:
    contamination: float
    precision: float
    recall: float
    f1: float
    n_normal: int
    n_outliers: int


def _make_labeled_transactions(
    n_normal: int = 90,
    n_outliers: int = 10,
    normal_mean: float = 250.0,
    normal_std: float = 40.0,
    outlier_multiplier: float = 12.0,
    seed: int = 7,
) -> tuple[list[dict], list[bool]]:
    """Build synthetic transactions with known ground-truth outlier labels.

    Normal amounts are drawn from a tight Gaussian band (typical recurring
    transaction sizes); outliers are injected at a large multiple of the
    mean, mimicking the kind of value a fat-fingered decimal or fraudulent
    entry would produce.
    """
    rng = random.Random(seed)
    transactions: list[dict] = []
    labels: list[bool] = []  # True == actually an outlier

    for i in range(n_normal):
        amount = max(1.0, rng.gauss(normal_mean, normal_std))
        transactions.append({"amount": round(amount, 2), "page": 1, "context": f"normal-{i}"})
        labels.append(False)

    for i in range(n_outliers):
        amount = normal_mean * outlier_multiplier * rng.uniform(0.8, 1.3)
        transactions.append({"amount": round(amount, 2), "page": 1, "context": f"outlier-{i}"})
        labels.append(True)

    # Shuffle transactions and labels together so ordering isn't a signal.
    paired = list(zip(transactions, labels, strict=True))
    rng.shuffle(paired)
    transactions, labels = (list(t) for t in zip(*paired, strict=True))
    return transactions, labels


def evaluate(contamination: float, seed: int = 7) -> EvalResult:
    transactions, labels = _make_labeled_transactions(seed=seed)
    _normal, flagged = detect_anomalies(transactions, contamination=contamination)

    flagged_contexts = {item["context"] for item in flagged}
    true_positives = 0
    false_positives = 0
    n_actual_outliers = sum(labels)

    for txn, is_outlier in zip(transactions, labels, strict=True):
        was_flagged = txn["context"] in flagged_contexts
        if was_flagged and is_outlier:
            true_positives += 1
        elif was_flagged and not is_outlier:
            false_positives += 1

    precision = true_positives / len(flagged) if flagged else 0.0
    recall = true_positives / n_actual_outliers if n_actual_outliers else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return EvalResult(
        contamination=contamination,
        precision=round(precision, 3),
        recall=round(recall, 3),
        f1=round(f1, 3),
        n_normal=len(transactions) - n_actual_outliers,
        n_outliers=n_actual_outliers,
    )


@pytest.mark.parametrize("contamination", [0.08, 0.1, 0.12, 0.15])
def test_detector_beats_minimum_bar_at_default_and_nearby_settings(contamination: float) -> None:
    """Regression guard: contamination in [0.08, 0.15] must clear F1 >= 0.6
    on this synthetic set (10% true outlier rate, 12x-mean outlier size).
    This is a floor, not a target -- see benchmarks/results.md for the full
    sweep (0.05-0.25) and the rationale for the 0.1 default. Values outside
    this range (0.05, 0.2, 0.25) measurably underperform and are
    intentionally excluded from the regression bar rather than papered over.
    """
    result = evaluate(contamination)
    assert result.f1 >= 0.6, (
        f"contamination={contamination}: F1={result.f1} "
        f"(precision={result.precision}, recall={result.recall})"
    )


def test_default_contamination_matches_true_outlier_rate_reasonably_well() -> None:
    """The dataset's true outlier rate is 10%; the configured default
    (settings.anomaly_contamination == 0.1) should be close to that, since
    IsolationForest's contamination parameter is most accurate when it
    approximates the true anomaly proportion."""
    from financial_doc_tool.config import settings

    assert 0.05 <= settings.anomaly_contamination <= 0.15


if __name__ == "__main__":
    print(f"{'contamination':>13} | {'precision':>9} | {'recall':>6} | {'f1':>5}")
    print("-" * 46)
    for c in (0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25):
        r = evaluate(c)
        print(f"{r.contamination:>13} | {r.precision:>9} | {r.recall:>6} | {r.f1:>5}")
