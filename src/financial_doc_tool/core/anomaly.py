from __future__ import annotations

import re
from typing import Any

import pandas as pd
from sklearn.ensemble import IsolationForest

from financial_doc_tool.config import settings

# Matches amounts either comma-grouped ("$12,500.00") or plain digit runs
# ("$12500.00"). The original pattern only matched \d{1,3} with mandatory
# comma-grouping beyond that, which silently truncated any un-grouped
# amount over 3 digits (e.g. "$12500.00" parsed as "125" + a bogus "00.00")
# -- a real accuracy bug, since plain-digit amounts without thousands
# separators are common in exported financial text. See
# tests/unit/test_anomaly.py::test_extract_transactions_handles_ungrouped_large_amounts.
_AMOUNT_PATTERN = re.compile(r"[\$\u20B9]?\s*(\d{1,3}(?:,\d{3})+(?:\.\d{2})?|\d+(?:\.\d{2})?)")


def extract_transactions(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract currency-like values from chunk text."""
    transactions: list[dict[str, Any]] = []

    for chunk in chunks:
        amounts = _AMOUNT_PATTERN.findall(chunk["content"])
        for amount in amounts:
            value = float(amount.replace(",", ""))
            if value > 0:
                transactions.append(
                    {
                        "amount": value,
                        "page": chunk["page"],
                        "context": chunk["content"][:100],
                    }
                )

    return transactions


def detect_anomalies(
    transactions: list[dict[str, Any]],
    contamination: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split transactions into normal and flagged groups via IsolationForest.

    contamination defaults to settings.anomaly_contamination (previously a
    hardcoded 0.1 with no stated justification). See benchmarks/results.md
    for the precision/recall this default achieves against a labeled
    synthetic evaluation set, and tests/performance/test_anomaly_eval.py for
    the evaluation harness itself.
    """
    contamination = contamination if contamination is not None else settings.anomaly_contamination
    if len(transactions) < 5:
        return transactions, []

    frame = pd.DataFrame(transactions)
    amounts = frame[["amount"]].values
    model = IsolationForest(contamination=contamination, random_state=42)

    frame["anomaly"] = model.fit_predict(amounts)

    normal = frame[frame["anomaly"] == 1].to_dict("records")
    anomalies = frame[frame["anomaly"] == -1].to_dict("records")
    return normal, anomalies
