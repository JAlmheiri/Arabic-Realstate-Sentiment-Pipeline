# runs Arabic comments through local Ollama LLM, extracts
# topic/sentiment, checks against ground truth

import pandas as pd
import ollama
import time

MODEL = "llama3.2"

PROMPT_TEMPLATE = """You are analyzing an Arabic real estate comment. 
Given the following Arabic text, respond ONLY in this exact JSON format with no extra text:
{{
  "topic_en": "<main topic in English, e.g. property prices, loans, government policy, housing quality>",
  "topic_ar": "<main topic in Arabic, e.g. أسعار العقارات, القروض, السياسة الحكومية, جودة السكن>",
  "sentiment": "<Pos, Neg, or Mix>",
  "price_mentioned": "<any price or number mentioned, or null>"
}}

Arabic text: {text}"""

def analyze_comment(text: str) -> dict:
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}]
        )
        raw = response["message"]["content"].strip()

        # parsing JSON response
        import json, re
        # stripping any markdown code fences the model might add
        raw = re.sub(r"```json|```", "", raw).strip()
        raw = raw.replace("،", ",")  # replacing Arabic comma (consistent issue)
        result = json.loads(raw)
        return {
            "llm_topic_en": result.get("topic_en", None).lower().strip(),
            "llm_topic_ar": result.get("topic_ar", None).strip(),
            "llm_sentiment": result.get("sentiment", None),
            "llm_price_mentioned": result.get("price_mentioned", None),
            "llm_error": None
        }
    except Exception as e:
        return {
            "llm_topic_en": None,
            "llm_topic_ar": None,
            "llm_sentiment": None,
            "llm_price_mentioned": None,
            "llm_error": str(e)
        }

def transform(df: pd.DataFrame, sample_size: int = 50) -> pd.DataFrame:
    # sample to keep runtime manageable on my local hardware
    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    print(f"  Processing {sample_size} comments through local LLM...")

    results = []
    for i, row in df.iterrows():
        if i % 10 == 0:
            print(f"  Progress: {i}/{sample_size}")
        result = analyze_comment(row["text"])
        results.append(result)
        time.sleep(0.5)  # to avoid overwhelming local model

    llm_df = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), llm_df], axis=1)

    # data quality check: does LLM sentiment match ground truth?
    df["sentiment_match"] = df["polarity"] == df["llm_sentiment"]

    # ingestion timestamp
    df["ingested_at"] = pd.Timestamp.utcnow()

    return df

def validate(df: pd.DataFrame) -> None:
    print(f"\n  Total processed: {len(df)}")
    print(f"  LLM errors: {df['llm_error'].notna().sum()}")
    print(f"  Sentiment match rate: {df['sentiment_match'].mean():.1%}")
    print(f"  Match breakdown:\n{df.groupby('polarity')['sentiment_match'].mean().round(2)}")
    print(f"  Topics found:\n{df['llm_topic_en'].value_counts().head(5)}")