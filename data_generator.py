import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_transactions():
    n = 10000
    np.random.seed(42)
    customer_ids = np.random.choice(range(1, 1001), n)
    amounts = np.random.gamma(2, 50, n).astype(int)
    timestamps = [datetime.now() - timedelta(seconds=np.random.randint(0, 7*24*3600)) for _ in range(n)]
    df = pd.DataFrame({
        'txn_id': range(1, n+1),
        'customer_id': customer_ids,
        'amount': amounts,
        'timestamp': timestamps,
        'merchant': np.random.choice(['Amazon', 'Walmart', 'Target', 'BestBuy'], n)
    })
    df['risk_label'] = ((df['amount'] > 200) & (df['merchant'] == 'BestBuy')).astype(int)
    return df

if __name__ == "__main__":
    df = generate_transactions()
    df.to_csv('transactions.csv', index=False)
    print("✅ transactions.csv generated")