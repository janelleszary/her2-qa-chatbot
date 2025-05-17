# HER2 QA Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers scientific questions based on the 1987 Slamon et al. paper on HER-2/neu amplification in breast cancer.

Built with Python, LangChain, and Ollama to support fast local inference and reproducible evaluation.

Project created by Janelle Szary, May 2025.

---

## Quickstart

### 1. Install Ollama

Ollama runs open-source LLMs locally. Follow installation instructions at https://ollama.com, then start the model:

```bash
ollama run mistral
```

### 2. Clone and set up environment

```bash
git clone https://github.com/your-repo-url
cd her2-qa-chatbot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

### 3. Run the chatbot

```bash
python qa_chatbot.py
```

---

## Project Overview

This system:
- Ingests and processes scientific text into searchable semantic chunks
- Embeds using multiple biomedical and general-purpose models
- Retrieves relevant information using vector similarity
- Generates answers using a local or fallback LLM

Includes a full evaluation pipeline for retrieval precision, recall, F1, and hallucination resistance.

---

## Features

- Local LLM generation with Mistral (via Ollama)
- Biomedical-specific retriever benchmarking (`e5`, `PubMedBERT`, `BioBERT`, `SBERT`)
- Robust chunking and preprocessing
- Manual and automated answer evaluation tools

---

## Evaluation Methodology

Evaluation set: `eval_dataset.jsonl` with 9 manually curated biomedical QA pairs.

Metrics:
- Recall@k, precision@k, F1
- Unique chunk recall (corrected)
- Answer similarity (token overlap, character similarity, cosine similarity)

Best model: `e5` at `k=3`, `threshold=0.70` showed the best tradeoff between coverage and redundancy. `PubMedBERT` had higher top-1 accuracy but hallucinated more in no-answer cases.

---

## Development Notes

### PDF Ingestion

Due to poor OCR from scientific PDFs, the transcript was manually cleaned and citations removed. Only the text was included—figures and tables were excluded.

### Chunking Strategy

- Paragraphs are split into sentences
- Chunks are formed by merging sentences up to a token limit (default: 100 tokens with 20-token overlap)
- Smaller, dense chunks performed better for biomedical QA

### Embedding Models Considered

- `e5` – most effective overall
- `PubMedBERT` – good for top-1 accuracy
- `SBERT` – solid fallback
- `BioBERT` – underperformed in precision



---

## Hardware & Compatibility

Ollama requires:
- macOS (Apple Silicon or Intel with Rosetta)
- Windows (with WSL2)
- Linux with AVX support

---

## License

MIT License

---

## Acknowledgments

- Slamon et al. (1987) for the source paper
- Hugging Face & LangChain for open-source tools
- Ollama for local LLM inference
