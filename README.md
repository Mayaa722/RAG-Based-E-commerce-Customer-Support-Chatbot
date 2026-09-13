# RAG-Based E-commerce Customer Support Chatbot

A fully integrated, end-to-end NLP pipeline for e-commerce customer support: every message is
routed through **language detection → sentiment/emotion classification → intent classification
→ (RAG generation | direct reply | escalation)**.

## Project layout

```
project/
├── README.md                     
├── requirements.txt
├── build_notebooks.py             <- (re)generates the notebooks below from source
├── data/                          <- small bundled sample CSVs (for offline dev/testing;
│                                      real datasets load automatically when available)
│   ├── sample_language_id.csv
│   ├── sample_emotion.csv
│   └── sample_support_kb.csv
├── notebooks/                     <- the 4 required deliverable notebooks
│   ├── 01_language_detection.ipynb
│   ├── 02_sentiment_classifier.ipynb
│   ├── 03_intent_classifier.ipynb
│   └── 04_rag_pipeline.ipynb
├── src/                           <- shared, importable implementation (used by both the
│   │                                  notebooks and the deployment app, so there's exactly
│   │                                  one source of truth per module)
│   ├── language_detection.py      <- Module 1
│   ├── sentiment_classifier.py    <- Module 2
│   ├── intent_classifier.py       <- Module 3
│   ├── intent_mapping.py          <- 27-intent -> 7-bucket routing taxonomy
│   ├── rag_pipeline.py            <- Module 4
│   ├── pipeline.py                <- orchestration: ties all 4 modules + routing together
│   └── paths.py                   <- cwd-independent path helpers
├── app/
│   └── main.py                    <- FastAPI deployment (`/chat`, `/health`)
├── scripts/
│   └── train_all.py               <- trains + persists all 4 models to models/
└── models/                        <- trained model artifacts land here (git-ignored)
```

## Quickstart

```bash
pip install -r requirements.txt

# optional: put real credentials in your shell env for full functionality
export GROQ_API_KEY=...          # https://console.groq.com (free tier)
# export QDRANT_URL=...          # only if using QdrantStore instead of local FAISS
# export QDRANT_API_KEY=...

# 1) train and persist all 4 models
python -m scripts.train_all

# 2) run the deployment API
uvicorn app.main:app --reload --port 8000

# 3) chat
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Where is my order? It never arrived and I am really frustrated."}'
```

If real dataset/API access (HuggingFace, Groq, Qdrant) isn't available in your environment,
**every module automatically falls back to the bundled sample data and offline/extractive
behavior**, clearly logged with a `[module_name] Falling back to ...` message, so the whole
pipeline (notebooks, training script, and API) is runnable end to end regardless.

## System overview

Every customer message passes through four stages before a final response is produced:

1. **Language Detection** – identifies the message language (so we search the right KB
   entries and reply in-language).
2. **Sentiment / Emotion Classification** – detects frustration vs. neutral vs. satisfied tone.
3. **Intent Classification** – routes to the correct handling path.
4. **Q&A RAG** – retrieves grounded knowledge-base context and generates the answer.

## Module summaries & key design decisions

### 1) Language Detection
- **Dataset:** `papluca/language-identification` (90k rows, 20 languages).
- **Approach:** character n-gram (2–5) TF-IDF + `LinearSVC` (calibrated for confidence scores).
- **Why char n-grams, not word-level:** language identity is a sub-word signal (letter
  combinations, diacritics) and works even for languages without whitespace tokenization
  (Chinese, Japanese, Korean).
- **Enhancement:** a confidence gate — low-confidence predictions fall back to `en` rather than
  confidently guessing, since a wrong language breaks downstream KB retrieval.

### 2) Sentiment / Emotion Classifier
- **Dataset:** `dair-ai/emotion` (~20k Twitter messages, 6 emotions).
- **Approach:** **BiLSTM** (RNN option from the brief), chosen over a transformer for
  CPU-trainability and full transparency for the oral assessment. A transformer fine-tune is
  the documented upgrade path if GPU time and higher accuracy are wanted.
- **Label mapping:** 6 fine emotions trained → mapped to 3 routing buckets:
  `negative={sadness,anger,fear}`, `positive={joy,love}`, `neutral={surprise}`.
- **Domain shift:** the dataset is Twitter text, not support text. `data/sample_emotion.csv` is
  a small hand-written customer-support-style set used as an independent qualitative check
  (see notebook §2.7), not for training.
- **Offline fallback:** if PyTorch isn't installed, `train()` automatically trains a
  TF-IDF + Logistic Regression model instead, clearly logged as a fallback — the graded
  deliverable remains the BiLSTM.

### 3) Intent Classifier
- **Dataset:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (26,872 rows, 27
  intents, gold-labeled).
- **Approach:** supervised **word-level TF-IDF + `LinearSVC`** (`class_weight="balanced"`)
  directly on the gold `intent` column — genuine supervised classification, not zero/few-shot,
  since labels already exist (per the brief's recommendation).
- **Two-stage labeling:** trained/evaluated on the 27 fine-grained gold labels, then mapped
  deterministically (`src/intent_mapping.py`) to 7 coarse routing buckets:

  | Coarse bucket | Fine intents | Routing behavior |
  |---|---|---|
  | `greeting_goodbye_gratitude` | greet, bye, thank, ... | direct reply, no RAG |
  | `order_status` | track_order, delivery_options, delivery_period | RAG |
  | `order_management` | cancel_order, change_order, place_order | RAG |
  | `billing_and_refunds` | check_invoice, get_refund, payment_issue | RAG |
  | `account_management` | create_account, edit_account, delete_account, switch_account, recover_password | RAG |
  | `complaint` | complaint, review | **escalate** (priority, human review) |
  | `out_of_scope` | out_of_scope | direct reply (can't-help message) |

### 4) Q&A RAG Pipeline
- **Knowledge base:** same Bitext dataset — `instruction` (question) embedded for retrieval,
  paired `response` (answer) injected as grounding context, per the brief's guidance.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector store:** local **FAISS** (`IndexFlatIP` over normalized vectors = cosine similarity)
  by default, to avoid an external account dependency; a `QdrantStore` adapter with the
  identical `.search()` interface is included as a drop-in swap for a persistent/shared
  deployment.
- **LLM:** **Groq**, `openai/gpt-oss-120b` (default) / `openai/gpt-oss-20b` (fallback), called
  with the exact system/user prompt template specified in the brief — including injecting the
  detected sentiment bucket so the model acknowledges frustration before answering, and an
  explicit instruction to admit when retrieved context doesn't cover the question and offer
  escalation rather than guessing.

## Routing / escalation design decision

The brief asks us to document whichever choice we make for handling complaint / negative
sentiment messages, and why. **We escalate on the `complaint` coarse intent specifically (not
on negative sentiment alone).**

- A genuine complaint/review (`complaint`, `review` fine intents) is qualitatively different
  from a frustrated customer asking a normal support question (e.g. *"I'm furious my refund is
  late, where is it?"* is still `billing_and_refunds` intent, just negative tone). Auto-
  generating a reply to an actual complaint risks sounding dismissive or making commitments the
  company can't keep — so these are flagged for **human review** with a holding message
  instead of an LLM-generated response.
- For any *other* intent bucket with negative sentiment, we keep **auto-response** but the RAG
  system prompt is instructed to acknowledge the customer's frustration before answering (via
  the `{detected_sentiment}` slot) — tone-adjusted, but still self-served.

This two-tier design satisfies the brief's requirement to "route complaint/negative-sentiment
messages distinctly" while keeping routine (if frustrated) questions from unnecessarily
bottlenecking on human agents.

## Running the notebooks

```bash
jupyter notebook notebooks/
```

Each notebook is self-contained: it imports from `src/`, loads data (real HF dataset or bundled
sample fallback), trains, evaluates, runs inference demos, and saves the model to `models/`.
`build_notebooks.py` regenerates all four from scratch if you want to see/modify the generation
script itself (all cells were executed in this environment with **zero errors** prior to
delivery — see notebook outputs).

## Deployment API

`app/main.py` (FastAPI) exposes:
- `GET /health` — readiness check.
- `POST /chat` — `{"message": "..."}` → full pipeline output: final `answer`, plus the
  intermediate `language`, `sentiment`, `intent`, `routing_behavior`, and `retrieved_context`
  for transparency/debugging.

On startup it loads persisted models from `models/` (run `scripts/train_all.py` first for fast
startup) or trains fresh ones on the fly otherwise.

## Known limitations / honest caveats

- This sandbox's network access does not include huggingface.co, groq.com, or Qdrant Cloud, so
  every module here was developed and verified end-to-end against small bundled sample CSVs
  (`data/*.csv`), not the full real datasets. All four notebooks were executed with **zero
  errors** on this sample data. The code paths for the real datasets/APIs (`datasets.load_dataset`,
  `sentence-transformers`, `groq`, `qdrant-client`) are implemented and will be exercised
  automatically the first time you run this with normal internet access and your own API keys
  — but their numeric results (accuracy, retrieval quality) have not been verified against the
  full-scale data by us, only the code's correctness/logic. Please re-run each notebook / the
  training script once with real access before your assessment so all reported metrics are the
  real ones.
- PyTorch could not be installed in this sandbox (disk constraints), so Module 2's BiLSTM path
  was verified by static review + syntax/logic testing of every non-torch helper, not by an
  actual training run. Its automatic TF-IDF/LogReg fallback *was* executed end to end. Please
  run `python -m src.sentiment_classifier` (or the notebook) yourself with torch installed
  before the assessment to confirm training converges as expected and to get real numbers to
  discuss.
- The bundled sample CSVs are deliberately tiny (20–26 rows) — just enough to prove every code
  path executes correctly. They are not meant to produce meaningful accuracy numbers; train
  against the real HF datasets for actual reported metrics.
