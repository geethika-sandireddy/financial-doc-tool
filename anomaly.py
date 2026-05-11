import re
from typing import Any

import pandas as pd
from sklearn.ensemble import IsolationForest


def extract_transactions(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract currency-like values from chunk text."""
    transactions: list[dict[str, Any]] = []
    amount_pattern = re.compile(r"[\$\u20B9]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")

    for chunk in chunks:
        amounts = amount_pattern.findall(chunk["content"])
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split transactions into normal and flagged groups."""
    if len(transactions) < 5:
        return transactions, []

    frame = pd.DataFrame(transactions)
    amounts = frame[["amount"]].values
    model = IsolationForest(contamination=0.1, random_state=42)

    frame["anomaly"] = model.fit_predict(amounts)

    normal = frame[frame["anomaly"] == 1].to_dict("records")
    anomalies = frame[frame["anomaly"] == -1].to_dict("records")
    return normal, anomalies
