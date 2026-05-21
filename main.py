# full orch arabic real estate sentiment ETL pipeline.
# extract → transform → load

from src.extract import extract
from src.transform import transform, validate
from src.load import load

def run_pipeline():
    print("=== Arabic Real Estate Sentiment Pipeline ===\n")

    print("[1/3] Extracting...")
    raw_df = extract()

    print("\n[2/3] Transforming & Analyzing...")
    clean_df = transform(raw_df, sample_size=50)
    validate(clean_df)

    print("\n[3/3] Loading...")
    load(clean_df)

    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    run_pipeline()