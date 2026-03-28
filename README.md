# DR-RAG: Addressing Retrieval Misalignment in Low-Resource Urdu Question Answering

> **Accepted at CHiPSAL 2026** — Challenges in Processing South Asian Languages, co-located with LREC 2026

## Overview

DR-RAG (Dual-Representation Retrieval-Augmented Generation) is a linguistically motivated RAG framework designed to address retrieval misalignment in morphologically rich, low-resource languages — with Urdu as the primary target.

Standard RAG pipelines assume well-calibrated dense embeddings and consistent tokenization, conditions that break down in Urdu due to heavy inflectional morphology, Nastaliq script inconsistencies, and limited training data. DR-RAG addresses this by indexing each document in two complementary forms: overlapping text chunks and automatically generated question-answer (QA) pairs. A confidence-aware fallback mechanism routes queries to the QA index first, switching to chunk-based retrieval only when confidence falls below a set threshold.

---

## Key Results

| Metric | Traditional RAG | MultiVector | DR-RAG |
|---|---|---|---|
| METEOR (Urdu) | 0.0056 | 0.2555 | **0.2134** |
| ROUGE-1 (Urdu) | 0.0660 | 0.1569 | **0.1584** |
| BERTScore F1 (Urdu) | 0.7955 | 0.8057 | **0.8134** |
| Faithfulness (Urdu) | 0.8150 | 0.8480 | **0.8760** |
| Generation Latency (Urdu) | 14.46s | 1.47s | **8.30s** |
| Context Precision (Urdu) | 0.0311 | 0.0010 | **0.0672** |
| Context Recall (Urdu) | 0.2880 | 0.0040 | **0.4023** |

- **38× METEOR improvement** over Traditional RAG on Urdu
- **140% ROUGE-1 gain** over Traditional RAG on Urdu
- **43% latency reduction** in Urdu generation
- **60× higher retrieval precision** over MultiVector on Urdu
- English performance remains **competitive** throughout

---

## System Architecture

DR-RAG operates in three phases:

**Phase 1 — Document Processing & Dual Indexing**
- Documents are split into overlapping chunks (200–500 tokens) using a recursive character splitter
- An LLM generates multiple QA pairs from each chunk
- Both chunks and QA pairs are embedded using language-specific models and stored in separate Qdrant indices

**Phase 2 — Confidence-Aware Retrieval**
- Incoming queries are first matched against the QA index
- If the top similarity score ≥ τ = 0.80 → use QA context
- If score < τ = 0.80 → fall back to chunk-based retrieval

**Phase 3 — Answer Generation**
- Retrieved context is passed to a generative model for final answer synthesis

---

## Repository Structure

```
DR_RAG/
│
├── Main_Pipeline_Urdu.ipynb          # Full DR-RAG pipeline for Urdu
├── Main_Pipeline_English.ipynb       # Full DR-RAG pipeline for English
│
├── Urdu_QA_Generation.ipynb          # QA pair generation from Urdu PDFs (GPT-4o)
├── English_QA_Generation.ipynb       # QA pair generation from English PDFs (LLaMA 2 7B)
│
├── NLP_Evaluation_Urdu.ipynb         # BLEU, ROUGE, METEOR, BERTScore for Urdu
├── NLP_Evaluation_English.ipynb      # BLEU, ROUGE, METEOR, BERTScore for English
├── NLP_Evaluation_Urdu_multivector.ipynb  # Same metrics for MultiVector baseline
│
├── RAGAS_Evaluation_Urdu.ipynb       # RAGAS metrics (Faithfulness, Relevancy, Recall)
├── RAGAS_Evaluation_English.ipynb    # RAGAS metrics for English
├── RAGAS_evaluation_multivector.ipynb     # RAGAS for MultiVector baseline
│
├── llm_evaluation.ipynb              # LLM-as-Judge evaluation (GPT-3.5 Turbo)
│
├── urdu_pdfs/                        # Urdu source documents (UQA-aligned)
│   ├── 2008 Earthquack.pdf
│   └── 2008 Olympics.pdf
│
└── english_pdfs/                     # English source documents (SQuAD 2.0-aligned)
    ├── Adult_contemporary_music.pdf
    ├── Airport.pdf
    ├── Aircraft_carrier.pdf
    ├── Affirmative_action_in_the_United_States.pdf
    └── 2008_Summer_Olympics_torch_relay.pdf
```

---

## Models Used

| Component | English | Urdu |
|---|---|---|
| QA Generation | LLaMA 2 7B | GPT-4o |
| Embedding Model | `all-MiniLM-L6-v2` | `intfloat/multilingual-e5-large` |
| Answer Generation | LLaMA3-ChatQA 8B | LLaMA 3.1 8B |
| Vector Database | Qdrant (local) | Qdrant (local) |
| LLM-as-Judge | GPT-3.5 Turbo | GPT-3.5 Turbo |

---

## Datasets

- **English:** [SQuAD 2.0](https://arxiv.org/abs/1806.03822) — 1,200 queries, 5 indexed documents
- **Urdu:** [UQA](https://arxiv.org/abs/2405.01458) — 900 queries, 2 indexed documents

---

## Installation

```bash
pip install transformers torch sentencepiece pdfminer.six numpy pandas tqdm \
            requests sentence-transformers langchain qdrant-client openai \
            nltk sacrebleu evaluate bert-score PyMuPDF stanza
```

---

## Usage

### Step 1 — Generate QA Pairs

Run `Urdu_QA_Generation.ipynb` or `English_QA_Generation.ipynb` to generate QA pairs from your PDF documents. Set your OpenAI API key for Urdu generation.

### Step 2 — Run the Main Pipeline

Run `Main_Pipeline_Urdu.ipynb` or `Main_Pipeline_English.ipynb`. This will:
1. Extract and chunk your PDFs
2. Embed chunks and QA pairs separately
3. Index both into Qdrant
4. Run queries with confidence-aware fallback
5. Generate answers using the language-appropriate LLM

### Step 3 — Evaluate

- **NLP Metrics** (BLEU, ROUGE, METEOR, BERTScore): Run the corresponding `NLP_Evaluation_*.ipynb`
- **RAGAS Metrics** (Faithfulness, Relevancy, Recall): Run the corresponding `RAGAS_Evaluation_*.ipynb`
- **LLM-as-Judge**: Run `llm_evaluation.ipynb` with your OpenAI key

> ⚠️ **API Keys:** Replace placeholder API keys (`sk-proj--`) in the notebooks with your own OpenAI API key before running.

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{drrag2026,
  title     = {DR-RAG: Addressing Retrieval Misalignment in Low-Resource Urdu Question Answering},
  author    = {Saad Ahmad and Muhammad Hammad and Faizad Ullah and Asim Karim},
  booktitle = {Proceedings of CHiPSAL 2026, co-located with LREC 2026},
  year      = {2026}
}
```

---

## License

This project is released for research purposes. The datasets used (UQA and SQuAD 2.0) are publicly available and contain no personally identifiable information.
