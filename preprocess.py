import pandas as pd
import numpy as np
import random
from pathlib import Path


def generate_synthetic_dataset(data_path, n_meters=100, days=30):

    rows = []
    timestamps = pd.date_range(start="2025-01-01", periods=days*24, freq="h")

    for meter in range(n_meters):

        base = np.random.uniform(0.5, 2.0)
        is_theft_meter = random.random() < 0.3

        for t in timestamps:

            hour = t.hour

            if 6 <= hour <= 22:
                consumption = base + np.random.normal(0.2, 0.1)
            else:
                consumption = base * 0.5 + np.random.normal(0.1, 0.05)

            theft = 0

            if is_theft_meter and random.random() < 0.3:
                theft = 1
                if random.random() < 0.5:
                    consumption *= random.uniform(0.1, 0.4)
                else:
                    consumption *= random.uniform(1.8, 3.0)

            rows.append({
                "meter_id": meter,
                "timestamp": t,
                "consumption": max(consumption, 0),
                "theft": theft
            })

    df = pd.DataFrame(rows)
    Path(data_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)

    print("✅ Dataset generated")
    return df


def load_consumption_data(data_path):
    data_path = Path(data_path)

    if not data_path.exists():
        return generate_synthetic_dataset(data_path)

    return pd.read_csv(data_path)


def build_feature_matrix(df):

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day

    df['rolling_mean_24h'] = df.groupby('meter_id')['consumption'].transform(
        lambda x: x.rolling(24, min_periods=1).mean()
    )

    df['rolling_std_24h'] = df.groupby('meter_id')['consumption'].transform(
        lambda x: x.rolling(24, min_periods=1).std().fillna(0)
    )

    df['pct_change'] = df.groupby('meter_id')['consumption'].pct_change().fillna(0)

    df['night_consumption'] = (df['hour'] < 6).astype(int)

    df['high_peak_ratio'] = df['consumption'] / (df['rolling_mean_24h'] + 1e-5)

    features = [
        'consumption', 'hour', 'day',
        'rolling_mean_24h', 'rolling_std_24h',
        'pct_change', 'night_consumption', 'high_peak_ratio'
    ]

    X = df[features]
    y = df['theft']

    return X, y, features