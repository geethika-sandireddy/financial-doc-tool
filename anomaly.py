import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import re

def extract_transactions(chunks):
    transactions = []
    amount_pattern = re.compile(r'[\$₹]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)')
    
    for chunk in chunks:
        amounts = amount_pattern.findall(chunk['content'])
        for amount in amounts:
            clean = float(amount.replace(',', ''))
            if clean > 0:
                transactions.append({
                    'amount': clean,
                    'page': chunk['page'],
                    'context': chunk['content'][:100]
                })
    return transactions

def detect_anomalies(transactions):
    if len(transactions) < 5:
        return transactions, []

    df = pd.DataFrame(transactions)
    amounts = df[['amount']].values

    model = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly'] = model.fit_predict(amounts)

    anomalies = df[df['anomaly'] == -1].to_dict('records')
    normal = df[df['anomaly'] == 1].to_dict('records')

    return normal, anomalies