# RAG for APRA Regulatory Documents

This project builds and evaluates a RAG system for querying APRA regulatory documents, focusing on retrieval performance in long, structured financial text where relevant context is often difficult to locate.

## Example

**Query**
> What capital requirements do banks have to meet under APRA?

**Answer**
> Banks must meet APRA’s minimum capital requirements for Common Equity Tier 1 (CET1), Tier 1 Capital, and Total Capital, with capital targets typically set above these minima to provide buffers that absorb losses under stress. APRA’s framework is designed to ensure capital is “unquestionably strong,” with higher requirements for some banks and simplified requirements for smaller, less complex institutions.

*Answers include supporting source excerpts (“basis”), with conditions and limitations where relevant.*

## Evaluation

| Top K | Recall | MRR   | Hits      |
|-------|--------|-------|-----------|
| 3     | 0.767  | 0.651 | 115 / 150 |
| 5     | 0.833  | 0.666 | 125 / 150 |
| 10    | 0.893  | 0.675 | 134 / 150 |

*Evaluation over 150 synthetic queries generated from document chunks.*

## Usage

### Setup

Create a virtual environment (recommended) and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

### Pipeline

The pipeline is split into three stages:

1. **`ingest.py`**  
   Fetches APRA regulatory documents from `manifest.csv`, extracts structured section text from HTML, and splits documents into token-bounded chunks, writing results to `data/raw/documents.jsonl` and `data/processed/chunks.jsonl`.

2. **`index.py`**  
   Generates embeddings for each chunk and stores them with associated metadata in a persistent Chroma collection at `data/chroma`.

3. **`generate.py`**  
   Runs an interactive query loop. For each input, it embeds the query, retrieves relevant chunks via `retrieve.py`, constructs a context block, and generates an answer grounded in the retrieved text.

Run the pipeline in order:
```bash
python src/ingest.py
python src/index.py
python src/generate.py
```

### Configuration

Paths, models, and pipeline settings are in `src/config.py`.
