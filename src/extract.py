

# reads Arabic real estate CSV and normalizes columns
# source: Yafoz & Mouhoub (2020), IEEE SMC -> arabic sentiment analysis datasets

import pandas as pd
import os

DATA_PATH = os.path.join("data", "ArabicRealstateDataset.csv")

def extract() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = ["text", "polarity"]
    df["text"] = df["text"].str.strip()
    print(f"  Loaded {len(df)} records")
    print(f"  Polarity distribution:\n{df['polarity'].value_counts()}")
    return df