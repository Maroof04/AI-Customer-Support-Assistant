# 🤖 AI Customer Support Assistant

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-green.svg)](https://github.com/facebookresearch/faiss)
[![Test Suite](https://img.shields.io/badge/Pytest-100%25%20Passing-brightgreen.svg)](tests/)

An end-to-end, retrieval-augmented (RAG) AI Customer Support Assistant that classifies customer query **intent**, detects emotional **sentiment**, performs semantic **vector retrieval** over verified knowledge base articles using **FAISS**, and generates grounded, tone-adaptive answers using **Groq API** or an offline deterministic fallback engine.

Designed from first principles as an original, production-structured Machine Learning & NLP portfolio project.

---

## 📌 Project Overview

Traditional customer service chatbots either rely on rigid regex rules or hallucinate inaccurate answers with unconstrained LLMs. 

This project bridges that gap by implementing a **modular, multi-stage RAG pipeline**:
1. **Understands Intent**: Classifies customer queries across 6 core customer support intents (`refund`, `payment`, `delivery`, `cancellation`, `account_issue`, `technical_support`) using TF-IDF and calibrated Machine Learning.
2. **Detects Emotion & Tone**: Evaluates sentiment polarity (positive, neutral, negative) and emotional distress indicators to adapt the tone of the answer (e.g. empathetic de-escalation for frustrated customers vs. concise direct guidance for neutral inquiries).
3. **Retrieves Ground Truth with FAISS**: Generates dense semantic vector embeddings and executes cosine similarity search via FAISS (`IndexFlatIP`) to retrieve verified FAQ articles.
4. **Synthesizes Grounded Answers**: Couples retrieved documentation with optional Groq LLM inference (`llama-3.3-70b-versatile`), enforcing strict anti-hallucination guardrails and an automatic local fallback when no API key is set.
5. **Interactive Streamlit Web Interface**: Features chat history, live metadata inspection (intent confidence, sentiment, retrieved source docs), and a model evaluation dashboard.

---

## 🏗️ Architecture

```
                      ┌────────────────────────────┐
                      │    Customer Support Query   │
                      └──────────────┬─────────────┘
                                     │
                   ┌─────────────────┴─────────────────┐
                   ▼                                   ▼
        ┌─────────────────────┐             ┌─────────────────────┐
        │  Intent Classifier   │             │ Sentiment Analyzer  │
        │  (TF-IDF + Calibrated│             │ (VADER + Emotional  │
        │  Logistic Regression)│             │ Distress Detection) │
        └──────────┬──────────┘             └──────────┬──────────┘
                   │                                   │
                   │ Intent Tag & Confidence           │ Tone Guidance
                   │                                   │ (Empathetic / Direct / Warm)
                   ▼                                   │
        ┌─────────────────────┐                        │
        │    FAISS Retriever  │                        │
        │ (Dense Semantic VEC │                        │
        │   + IndexFlatIP)    │                        │
        └──────────┬──────────┘                        │
                   │ Top-k Grounded FAQ Articles       │
                   └─────────────────┬─────────────────┘
                                     ▼
                       ┌───────────────────────────┐
                       │ Context-Aware Generator   │
                       │  - Groq LLM (Optional)    │
                       │  - Grounded Local Engine  │
                       │  - Anti-Hallucination Gate│
                       └─────────────┬─────────────┘
                                     ▼
                       ┌───────────────────────────┐
                       │ Streamlit Chat Interface  │
                       │  - Interactive Dialogue   │
                       │  - Live Inspection Drawer │
                       │  - Evaluation Dashboard   │
                       └───────────────────────────┘
```

---

## ⚡ Key Features

- **🎯 Machine Learning Intent Classification**:
  - Predicts intent across 6 categories with probability calibration.
  - Built-in confidence thresholding (`CONFIDENCE_THRESHOLD = 0.35`): unconfident or out-of-scope queries trigger graceful escalation rather than random guesses.
  - Achieves **86.36% Accuracy** and **86.27% Weighted F1-Score** on held-out test data.

- **🎭 Sentiment & Emotional Tone Modulation**:
  - Leverages VADER NLP compound scoring and urgency heuristic analysis.
  - Dynamically modulates the assistant's persona:
    - **Negative / Frustrated**: Empathy-first apologies, validation, and de-escalation.
    - **Neutral**: Structured, concise, actionable step-by-step guidance.
    - **Positive**: Warm, appreciative confirmation.

- **⚡ FAISS Cosine Vector Search**:
  - Normalized dense embeddings paired with FAISS `IndexFlatIP`.
  - Mathematically equivalent to exact cosine similarity without runtime latency.
  - Flexible dual-backend: supports `SentenceTransformers` (`all-MiniLM-L6-v2`) when present, with a fast zero-dependency LSA dense vector fallback.

- **🛡️ Strict Anti-Hallucination & Fallback Engine**:
  - Grounded RAG ensures answers are sourced strictly from verified FAQ documentation.
  - If no relevant documentation exists, the assistant triggers a safe out-of-scope response rather than making up policies.
  - 100% functional without an API key through a local grounded generation engine.

- **🖥️ Full Streamlit Web UI**:
  - Multi-tab application: **Support Chatbot**, **Model Evaluation & Metrics**, and **Knowledge Base Browser**.
  - One-click quick sample queries for live demos.
  - Expandable drawer to inspect retrieved FAQ documents and confidence scores.

---

## 📊 Evaluation & Benchmarks

The Intent Classifier was trained and evaluated on held-out data using an 80/20 stratified split (264 training samples, 66 held-out test samples):

| Intent Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **cancellation** | **100.00%** | **100.00%** | **100.00%** | 11 |
| **technical_support** | **100.00%** | **90.91%** | **95.24%** | 11 |
| **account_issue** | **84.62%** | **100.00%** | **91.67%** | 11 |
| **refund** | **81.82%** | **81.82%** | **81.82%** | 11 |
| **delivery** | **80.00%** | **72.73%** | **76.19%** | 11 |
| **payment** | **72.73%** | **72.73%** | **72.73%** | 11 |
| **Overall Weighted Avg** | **86.53%** | **86.36%** | **86.27%** | **66** |

*All metrics and confusion matrix plots are automatically persisted to `models/evaluation_metrics.json` upon running the training script.*

---

## 📁 Repository Structure

```
ai-customer-support-assistant/
│
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── LICENSE                     # MIT License & attribution notice
├── .gitignore                  # Git ignore rules
├── .env.example                # Environment variables template
│
├── data/
│   ├── support_tickets.csv     # 330 curated customer support queries across 6 intents
│   ├── knowledge_base.csv      # 18 structured FAQ articles with verified resolutions
│   ├── build_tickets.py        # Ticket dataset generation script
│   └── build_kb.py             # Knowledge base builder script
│
├── models/
│   ├── intent_classifier.joblib # Serialized TF-IDF + Classifier pipeline
│   ├── faiss_index.bin         # Serialized FAISS vector index
│   ├── retriever_metadata.joblib # Knowledge base metadata payload
│   ├── evaluation_metrics.json # Test split evaluation results
│   ├── confusion_matrix.png    # Confusion matrix visualization
│   └── metrics_summary.png     # Metric comparison bar chart
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Text normalization, cleaning, contraction expansion
│   ├── intent_classifier.py    # TF-IDF vectorization & calibrated ML models
│   ├── sentiment_analyzer.py   # VADER sentiment & tone adaptation logic
│   ├── retriever.py            # FAISS vector similarity search
│   ├── response_generator.py   # Grounded RAG generator (Groq API + Local Fallback)
│   └── utils.py                # Pipeline orchestrator and plotting helpers
│
├── scripts/
│   └── train.py                # End-to-end training and evaluation script
│
└── tests/
    ├── conftest.py             # Pytest root configuration
    ├── test_preprocessing.py   # Preprocessing unit tests
    ├── test_intent_classifier.py# Classifier unit tests
    ├── test_sentiment_analyzer.py# Sentiment unit tests
    ├── test_retriever.py       # FAISS retrieval unit tests
    └── test_pipeline.py        # End-to-end integration tests
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Maroof04/AI-Customer-Support-Assistant.git
cd AI-Customer-Support-Assistant
```

### 2. Create and Activate Virtual Environment
```bash
# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Configure Groq API Key
If you want to enable live LLM generation:
```bash
cp .env.example .env
# Edit .env and insert your GROQ_API_KEY
```
*(Note: If no API key is provided, the application runs out of the box using its local grounded fallback engine!)*

---

## 🚀 Running the Project

### Step 1: Train the Models & Build FAISS Index
```bash
python scripts/train.py
```
This script will:
- Load the support ticket dataset and knowledge base.
- Split data into train (80%) and held-out test (20%) sets.
- Train the TF-IDF and calibrated classifier.
- Compute evaluation metrics and generate charts.
- Build and serialize the FAISS index into `models/`.

### Step 2: Run Unit & Integration Tests
```bash
python -m pytest tests -v
```
All 17 tests verify preprocessing, classifier predictions, sentiment tone mappings, vector retrieval, and RAG grounding.

### Step 3: Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📸 Application Preview

The Streamlit interface provides:
1. **Chat Window**: Natural back-and-forth conversation with real-time intent, sentiment, and source badges.
2. **Drawer Inspector**: Click *"Inspect Retrieved FAQ Source Document"* under any response to see the exact matched knowledge base article and similarity score.
3. **Evaluation Dashboard**: Live visual metrics showing accuracy, precision, recall, F1, and the confusion matrix.
4. **Knowledge Base Explorer**: Search through all 18 indexed support articles.

---

## 🎓 Technical & Architectural Decisions (Interview Deep-Dive)

### 1. Why TF-IDF + Calibrated Logistic Regression over a Black-Box Transformer?
For enterprise customer support intents with constrained class boundaries, classical ML with sublinear TF-IDF offers:
- **Low Latency**: Sub-millisecond inference compared to 50–200ms transformer forward passes.
- **Explainability**: Every n-gram feature has an interpretable weight in the classifier coefficients.
- **Calibrated Probabilities**: Gives realistic confidence estimates, allowing threshold-based fallback to human agents for out-of-scope questions.

### 2. Why IndexFlatIP for Cosine Similarity?
FAISS `IndexFlatIP` computes the inner product $\langle u, v \rangle$. By $L_2$-normalizing our embeddings ($\|u\|_2 = \|v\|_2 = 1$), the inner product is mathematically identical to cosine similarity:
$$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = u \cdot v$$
This avoids expensive square-root operations during search, yielding $O(N \cdot D)$ search performance.

### 3. How Are Hallucinations Prevented?
- The response generator enforces a strict system prompt instructing the model to answer exclusively from the provided knowledge base block.
- If the similarity score of the top retrieved article is below the threshold, the assistant politely declines and triggers a human escalation path rather than inventing an answer.

---

## 🔮 Future Improvements

- [ ] Add multilingual support using multilingual embeddings.
- [ ] Connect with ticketing platforms (e.g., Zendesk, Jira Service Desk API) for automatic ticket creation.
- [ ] Implement active learning to flag low-confidence user queries for human review and retraining.
- [ ] Add conversation session summary exports as PDF/CSV.

---

## 📄 License & Attribution

This project is licensed under the [MIT License](LICENSE).

**Attribution:**
Inspired by the technical concepts demonstrated in [jchaithra661-sys/ai-support-assistant](https://github.com/jchaithra661-sys/ai-support-assistant) (MIT License). Re-architected with calibrated ML intent classification, sentiment tone adaptation, FAISS vector indexing, and dual-backend RAG response generation.
