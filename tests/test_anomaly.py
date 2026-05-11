from anomaly import detect_anomalies, extract_transactions


def test_extract_transactions_parses_currency_values():
    chunks = [
        {"content": "Paid $1,200.50 to Vendor A and 350 to Vendor B", "page": 1},
        {"content": "Refund received: ₹2,000.00", "page": 2},
    ]

    transactions = extract_transactions(chunks)

    assert [item["amount"] for item in transactions] == [1200.50, 350.0, 2000.0]
    assert transactions[0]["page"] == 1


def test_detect_anomalies_returns_empty_flags_for_small_input():
    transactions = [
        {"amount": 100.0, "page": 1, "context": "a"},
        {"amount": 120.0, "page": 1, "context": "b"},
        {"amount": 140.0, "page": 1, "context": "c"},
    ]

    normal, anomalies = detect_anomalies(transactions)

    assert normal == transactions
    assert anomalies == []


def test_detect_anomalies_flags_outlier_in_larger_sample():
    transactions = [
        {"amount": 100.0, "page": 1, "context": "a"},
        {"amount": 102.0, "page": 1, "context": "b"},
        {"amount": 98.0, "page": 1, "context": "c"},
        {"amount": 101.0, "page": 1, "context": "d"},
        {"amount": 99.0, "page": 1, "context": "e"},
        {"amount": 5000.0, "page": 2, "context": "outlier"},
    ]

    normal, anomalies = detect_anomalies(transactions)

    assert len(normal) + len(anomalies) == len(transactions)
    assert any(item["amount"] == 5000.0 for item in anomalies)
