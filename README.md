# RAG for APRA Regulatory Documents

Most RAG examples assume clean, well-structured text. APRA regulatory documents are the opposite: long, repetitive, structurally consistent, and semantically dense.

This project explores how RAG performs on this kind of data.

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
   Generates embeddings for each chunk and stores them with associated metadata, in a persistent Chroma collection at `data/chroma`.

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
