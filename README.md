# RAG for APRA Regulatory Documents

This project builds and evaluates a RAG system for querying APRA regulatory documents, focusing on retrieval performance in long, structured financial text where relevant context is often difficult to locate.

## Example

**Query**

> What capital requirements do banks have to meet under APRA?

**Answer**

> Banks must meet APRA’s minimum capital requirements for Common Equity Tier 1 (CET1), Tier 1 Capital, and Total Capital, with capital targets typically set above these minima to provide buffers that absorb losses under stress. APRA’s framework is designed to ensure capital is “unquestionably strong,” with higher requirements for some banks and simplified requirements for smaller, less complex institutions.

*Answers include supporting source excerpts (“basis”), with conditions and limitations where relevant.*

## Evaluation

**Summary:** Hybrid search has the strongest ranking quality (MRR) at every top-k, and the best recall at top-k 5 and 10. Keyword search is competitive on recall at low top-k. Semantic search trails both.

### Keyword search


| Top-k | Recall | MRR   | Hits      |
| ----- | ------ | ----- | --------- |
| 3     | 0.760  | 0.590 | 114 / 150 |
| 5     | 0.840  | 0.608 | 126 / 150 |
| 10    | 0.893  | 0.612 | 134 / 150 |


### Semantic search


| Top-k | Recall | MRR   | Hits      |
| ----- | ------ | ----- | --------- |
| 3     | 0.687  | 0.552 | 103 / 150 |
| 5     | 0.740  | 0.564 | 111 / 150 |
| 10    | 0.840  | 0.578 | 126 / 150 |


### Hybrid search


| Top-k | Recall | MRR   | Hits      |
| ----- | ------ | ----- | --------- |
| 3     | 0.747  | 0.619 | 112 / 150 |
| 5     | 0.847  | 0.634 | 127 / 150 |
| 10    | 0.927  | 0.638 | 139 / 150 |


*Evaluated on 150 synthetic queries generated from document chunks.*

## Usage

### Setup

Install the project dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Create a `.env` file in the project root with your OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

### Pipeline

The pipeline is split into three stages:

1. **`ingest.py`**
  Discovers APRA regulatory documents from the Handbook sitemap and fetches each page. Writes HTML under `data/raw/html/` grouped by type (e.g. `faq/2465__fit-and-proper.html`) and a thin metadata index at `data/raw/documents.jsonl`. Skips the fetch if that index already exists.

2. **`index.py`**  
  Loads the stored HTML, extracts structured sections, splits them into token-bounded chunks at `data/processed/chunks.jsonl`, then embeds and stores them in Chroma (`data/chroma`) and BM25 (`data/bm25`).

3. **`generate.py`** 
  Runs an interactive query loop. For each input, it embeds the query, retrieves relevant chunks via `retrieve.py`, constructs a context block, and generates an answer grounded in the retrieved text.

Run the pipeline in order (from the project root). Pass `--env-file .env` so uv injects your API key:

```bash
uv run --env-file .env ingest
uv run --env-file .env index
uv run --env-file .env generate
```

Evaluation uses a fixed query set at `data/eval/queries.jsonl` so scores stay comparable across runs:

```bash
uv run --env-file .env evaluate
uv run --env-file .env evaluate --method hybrid --top-k 5
```

Defaults come from `config.py` (`SEARCH_METHOD`, `TOP_K`).

To intentionally rebuild that benchmark (invalidates prior scores):

```bash
uv run --env-file .env make-queries
```

To avoid repeating the flag, export `UV_ENV_FILE=.env` in your shell.

### Configuration

Paths, models, and pipeline settings are in `src/config.py`.
