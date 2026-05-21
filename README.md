# Arabic Real Estate Sentiment Pipeline
A local NLP pipeline that processes Arabic real estate comments using a locally run LLM (Ollama/Llama 3.2). 
It extracts topics bilingually and predicts sentiment, validated against ground truth labels. 
All processing runs locally; no data sent externally (testing private workflow).

## Architecture
CSV → extraction → transformation (Ollama LLM) → loading → SQLite

## Setup
```bash
pip install -r requirements.txt
# install Ollama from https://ollama.com, then:
ollama pull llama3.2
python main.py
```

## Dataset
6,434 Arabic real estate comments labeled Pos/Neg/Mix across 85 topics.  
Source: Yafoz & Mouhoub (2020), IEEE SMC. Available at: https://github.com/aymanya/Arabic-Sentiment-Analysis-Datasets  
Dataset not included; download and place in `data/`.

## Key Findings
- LLM associates price mentions with negativity, missing cases where falling prices are framed positively by writers
- Mix class not reliably detectable with a small local model; inherently ambiguous sentiment
- Some Arabic topic output garbled; known limitation of 3B parameter models on Arabic generation tasks
- JSON parse errors occur when Arabic text contains inline commas that conflict with JSON structure; partly resolved
