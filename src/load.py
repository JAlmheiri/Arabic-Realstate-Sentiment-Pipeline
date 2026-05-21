# loads transformed data into SQLite

import pandas as pd
from sqlalchemy import create_engine
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "arabic_realestate.db")

def load(df: pd.DataFrame) -> None:
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df.to_sql(
        name="realestate_sentiment",
        con=engine,
        if_exists="replace",
        index=False
    )
    print(f"  Loaded {len(df)} records into realestate_sentiment")
    print(f"  Database: {DB_PATH}")