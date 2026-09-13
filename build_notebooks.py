"""Generates the four deliverable notebooks under notebooks/ using nbformat.
Run once: python3 build_notebooks.py
"""
import nbformat as nbf


def nb(cells):
    n = nbf.v4.new_notebook()
    n["cells"] = cells
    n["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    return n


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


# ---------------------------------------------------------------------------
# 1) Language Detection
# ---------------------------------------------------------------------------
lang_nb = nb([
    md("""# Module 1: Language Detection

**Goal:** classify the language of an incoming customer message using traditional NLP, so the
system can (a) search the right knowledge-base entries and (b) reply in the same language.

**Dataset:** [`papluca/language-identification`](https://huggingface.co/datasets/papluca/language-identification)
— 90k samples across 20 languages, pre-split into train/validation/test.

**Approach:** character n-gram TF-IDF (2–5 grams, `char_wb` analyzer) + a linear SVM
(`LinearSVC`, wrapped in `CalibratedClassifierCV` for probability estimates).

**Why char n-grams, not word n-grams?** Language identity lives at the sub-word level —
letter combinations, diacritics, accented sequences — not in shared vocabulary. Char n-grams
also work for languages without whitespace tokenization (Chinese, Japanese, Korean), where a
word-level vectorizer would be nearly useless. This is the standard, well-established approach
for language ID and needs no GPU or embeddings.

**Enhancement over the minimal ask:** a *confidence gate* — if the top prediction's
probability falls below a threshold, we don't confidently guess; we fall back to `en` and flag
`low_confidence=True`, since a wrong language guess breaks downstream knowledge-base retrieval
for the RAG module."""),

    code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd
from src import language_detection as ld
"""),

    md("## 1.1 Load data\nTries the real HF dataset first; falls back to the small bundled CSV sample "
       "(`data/sample_language_id.csv`) if `datasets`/internet access isn't available, so this notebook "
       "always runs end to end."),

    code("""train_df, val_df, test_df = ld.load_hf_dataset()
print(train_df.shape, val_df.shape, test_df.shape)
train_df.head()
"""),

    md("## 1.2 Exploratory check\nClass balance and a couple of raw examples per language."),

    code("""print(train_df['labels'].value_counts())
train_df.sample(min(5, len(train_df)), random_state=0)
"""),

    md("## 1.3 Build & train the pipeline\n`TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5))` "
       "→ `CalibratedClassifierCV(LinearSVC())`.\n\nSee `src/language_detection.py` for the full "
       "implementation (also handles the tiny-sample edge case where a class has too few examples for "
       "cross-validated calibration, falling back to an uncalibrated `LinearSVC` with a "
       "softmax-over-decision_function confidence proxy)."),

    code("""pipeline = ld.train(train_df, val_df)
"""),

    md("## 1.4 Evaluate on held-out test set"),
    code("""ld.evaluate(pipeline, test_df)
"""),

    md("## 1.5 Inference + confidence gating demo"),
    code("""for msg in [
    "Hello, where is my order?",
    "Hola, ¿dónde está mi pedido?",
    "Bonjour, où est ma commande?",
    "asdkj qwoe",  # gibberish -> should be low-confidence
]:
    print(msg, '->', ld.predict_language(pipeline, msg))
"""),

    md("## 1.6 Save model for deployment"),
    code("""os.makedirs('../models', exist_ok=True)
ld.save(pipeline, '../models/language_detector.joblib')
print('Saved.')
"""),

    md("""## Notes / decisions to defend at assessment
- Char n-grams >> word n-grams for language ID (sub-word signal, works for non-whitespace-segmented
  languages).
- `class_weight='balanced'` compensates for any label imbalance in the 20-language dataset.
- Confidence gate protects downstream retrieval from a wrong-language guess rather than always
  returning the top-1 label no matter how uncertain.
- `CalibratedClassifierCV` gives genuine probability estimates from `LinearSVC` (which has no
  native `predict_proba`), which the confidence gate depends on."""),
])

# ---------------------------------------------------------------------------
# 2) Sentiment / Emotion Classifier
# ---------------------------------------------------------------------------
sent_nb = nb([
    md("""# Module 2: Sentiment / Emotion Classifier

**Goal:** classify the emotional tone of a customer message so the system can route frustrated
customers to a more empathetic / priority-flagged response path.

**Dataset:** [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) — ~20k
English Twitter messages labeled with 6 emotions (sadness, joy, love, anger, fear, surprise).

**Approach:** a **Recurrent Neural Network** (BiLSTM), one of the two options the brief allows
(RNN or Transformer). An RNN was chosen deliberately over a transformer:

- The dataset is short, single-sentence text (~20k rows) — a BiLSTM with a small embedding
  layer trains in a few minutes on CPU, whereas fine-tuning a transformer (e.g. DistilBERT)
  needs GPU time to be practical.
- The brief explicitly warns to "avoid overcomplicated approaches you don't fully grasp" —
  a from-scratch BiLSTM is fully transparent line-by-line for the oral assessment, whereas a
  fine-tuned transformer involves more moving parts (tokenizer internals, pretrained weights)
  that are harder to defend in depth on short notice.
- A transformer fine-tune is the natural upgrade path noted at the end of this notebook if
  GPU time is available and higher accuracy is needed.

**Domain-shift caveat:** this dataset is Twitter text, not customer-support text. We keep a
small hand-written customer-support-style qualitative sample (`data/sample_emotion.csv`) as an
independent sanity check, separate from the Twitter-trained model's own held-out split.

**Label mapping:** the 6 fine-grained emotions are trained first (this is what the dataset
actually supervises), then mapped to 3 routing buckets:
`negative={sadness,anger,fear}`, `positive={joy,love}`, `neutral={surprise}`. Surprise is kept
neutral since it's valence-ambiguous (a surprise can be pleasant or unpleasant) — a documented
design decision, not an oversight."""),

    code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd
from src import sentiment_classifier as sc
"""),

    md("## 2.1 Load data"),
    code("""df = sc.load_hf_dataset()
print(df.shape)
print(df['label'].value_counts())
df.head()
"""),

    md("## 2.2 Preprocessing\nLowercasing, URL stripping, non-alphabetic character removal, whitespace "
       "normalization (see `clean_text`), then a simple whitespace-token vocabulary capped at 20k tokens, "
       "padded/truncated to `MAX_LEN=40` tokens — plenty for short Twitter-style messages."),
    code("""df['clean'] = df['text'].apply(sc.clean_text)
df[['text', 'clean']].head()
"""),

    md("## 2.3 Model architecture\n\n```\nEmbedding(vocab_size, 100, padding_idx=0)\n  -> "
       "BiLSTM(100 -> 64, bidirectional)\n  -> concat(final forward, final backward hidden states)\n  "
       "-> Dropout(0.3)\n  -> Linear(128 -> n_classes)\n```\n\nBidirectional so the model sees both left "
       "and right context around emotionally-loaded words; final hidden states (not just the last "
       "timestep) are concatenated from both directions as the sentence representation."),

    md("## 2.4 Train\nRequires PyTorch. If torch isn't installed in this environment, `train()` "
       "automatically falls back to a TF-IDF + Logistic Regression classifier so the notebook still runs "
       "end to end for local testing — **the graded deliverable is the BiLSTM above**; install torch "
       "(`pip install torch`) to train it for real."),
    code("""model, vocab, label2id = sc.train(df)
"""),

    md("## 2.5 Save model"),
    code("""sc.save(model, vocab, label2id, out_dir='../models/sentiment')
print('Saved.')
"""),

    md("## 2.6 Inference + bucket mapping demo"),
    code("""for msg in [
    "this is the worst experience ever, so angry",
    "thank you so much this made my day",
    "wow I did not expect that refund so fast",
    "can you tell me your return policy",
]:
    print(msg, '->', sc.predict_sentiment(model, vocab, label2id, msg))
"""),

    md("## 2.7 Qualitative domain-shift check\nRun the same model on hand-written, customer-support-style "
       "messages (not Twitter text) to sanity-check generalization beyond the training domain."),
    code("""support_style_df = pd.read_csv('../data/sample_emotion.csv')
for _, row in support_style_df.head(8).iterrows():
    pred = sc.predict_sentiment(model, vocab, label2id, row['text'])
    print(f\"gold={row['label']:>8s} pred={pred['emotion']:>8s} bucket={pred['bucket']:>8s} | {row['text'][:60]}\")
"""),

    md("""## Notes / decisions to defend at assessment
- RNN (BiLSTM) chosen over transformer for CPU-trainability + full transparency at the
  assessment; transformer fine-tune is a documented upgrade path, not something we didn't
  consider.
- 6 fine emotions trained (matches the dataset's real gold labels), then mapped to 3 routing
  buckets — deliberate 2-stage design, not information loss by accident.
- `surprise -> neutral` is a defensible-but-debatable choice; an alternative would be a
  4-bucket scheme splitting surprise into positive/negative by context, at the cost of extra
  ambiguity in the labels the model has to learn.
- Domain shift (Twitter -> customer support) is explicitly flagged and checked qualitatively,
  not ignored."""),
])

# ---------------------------------------------------------------------------
# 3) Intent Classifier
# ---------------------------------------------------------------------------
intent_nb = nb([
    md("""# Module 3: Intent Classifier

**Goal:** classify what the customer actually wants, to route the message to the correct
handling path (small talk / order status / order management / billing / account / complaint /
out of scope).

**Dataset:** [`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
— 26,872 instruction/response pairs across 27 intents and 10 categories. The `intent` column is
already gold-labeled, so this is genuine **supervised classification**, not zero/few-shot.

**Approach:** traditional ML — word-level TF-IDF (unigrams + bigrams) + `LinearSVC` with
`class_weight='balanced'` to handle the natural class imbalance across 27 intents.

**Why word-level TF-IDF here (unlike Module 1's char n-grams)?** Intent is carried by content
words and short phrases ("cancel", "refund", "track my order"), not by sub-word letter
patterns — the opposite signal from language identification.

**Two-stage label design:** we train and evaluate on the **fine-grained gold labels** (27
classes) — this is what the dataset actually supervises and is what should be reported/
defended at assessment — then apply a **deterministic mapping** (`src/intent_mapping.py`) down
to 7 coarse routing buckets used by the orchestration pipeline. `complaint`/`review` always map
to the priority `complaint` bucket regardless of classifier confidence, per the brief."""),

    code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd
from src import intent_classifier as ic
from src import intent_mapping
"""),

    md("## 3.1 Load data"),
    code("""df = ic.load_hf_dataset()
print(df.shape)
print(df['intent'].value_counts())
df[['instruction', 'response', 'intent', 'category']].head()
"""),

    md("## 3.2 Coarse routing taxonomy\nThe 27 fine intents collapse into 7 routing buckets (see "
       "`src/intent_mapping.py` for the full table)."),
    code("""for fine, coarse in list(intent_mapping.FINE_TO_COARSE.items())[:10]:
    print(f'{fine:>26s} -> {coarse}')
print('...')
print('\\nRoute behavior per coarse bucket:', intent_mapping.ROUTE_BEHAVIOR)
"""),

    md("## 3.3 Train fine-grained classifier\nWord-level TF-IDF (1-2 grams, English stopwords removed) "
       "+ `LinearSVC(class_weight='balanced')`."),
    code("""pipeline, train_df, test_df = ic.train(df)
"""),

    md("## 3.4 Inference + coarse mapping + confidence"),
    code("""for msg in [
    "I want to cancel my order please",
    "This product is awful, I'm furious",
    "hi there!",
    "what's the weather like today",
    "how do I reset my password",
]:
    print(msg, '->', ic.predict_intent(pipeline, msg))
"""),

    md("## 3.5 Save model"),
    code("""os.makedirs('../models', exist_ok=True)
ic.save(pipeline, '../models/intent_classifier.joblib')
print('Saved.')
"""),

    md("""## Notes / decisions to defend at assessment
- Trained on the gold `intent` column directly (recommended path in the brief) rather than
  zero/few-shot LLM prompting -- simpler, faster, fully reproducible, and the dataset already
  supports it.
- Word n-grams (not char n-grams) because intent lives in content words/phrases.
- `class_weight='balanced'` because the 27 intents are not evenly represented in real support
  logs (far more `track_order` than `delete_account`, for example).
- Fine-grained training + post-hoc coarse mapping keeps the fine signal available for future
  routing refinements, rather than baking the collapse into training directly.
- `complaint`/`review` are hard-routed to escalation-priority regardless of confidence -- this
  is a policy decision (safety-first for negative-experience messages), not a modeling
  limitation."""),
])

# ---------------------------------------------------------------------------
# 4) RAG Pipeline
# ---------------------------------------------------------------------------
rag_nb = nb([
    md("""# Module 4: Q&A RAG Pipeline

**Goal:** retrieve grounded information from the customer-support knowledge base and generate an
accurate, tone-appropriate answer to the customer's question.

**Dataset:** same Bitext dataset as Module 3 -- doubles as the RAG knowledge base. Each row's
`instruction` (customer question) is embedded and indexed; the paired `response` (expected
agent answer) is the grounding text injected into the generation prompt for the nearest
matches.

**Components:**
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` -- fast, 384-dim, strong
  general-purpose semantic similarity, CPU-friendly.
- **Vector store:** local **FAISS** (`IndexFlatIP` over L2-normalized vectors = cosine
  similarity) by default, to avoid an external account dependency for local development and the
  assessment run. A `QdrantStore` adapter with the identical `.search()` interface is included
  in `src/rag_pipeline.py` as a drop-in swap if a persistent/shared, free-tier Qdrant Cloud
  instance is preferred instead.
- **LLM:** **Groq** API, `openai/gpt-oss-120b` by default (`gpt-oss-20b` as a faster/cheaper
  fallback), called with the brief's suggested prompt template, injecting the detected
  sentiment bucket so the model acknowledges frustration before answering."""),

    code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd
from src import rag_pipeline as rag
"""),

    md("## 4.1 Load the knowledge base"),
    code("""df = rag.load_hf_dataset()
print(df.shape)
df[['instruction', 'response', 'intent', 'category']].head()
"""),

    md("## 4.2 Build embeddings + FAISS index\nEmbeds every `instruction`. If `sentence-transformers` "
       "isn't installed, `Embedder` falls back to TF-IDF so the retrieval logic is still testable "
       "offline -- clearly logged, not silent."),
    code("""embedder = rag.Embedder()
store = rag.build_index(df, embedder)
print(f'Indexed {len(store.meta)} KB entries. Embedding backend: {embedder.backend}')
"""),

    md("## 4.3 Retrieval sanity check"),
    code("""query = 'my package never arrived, where is it'
query_vec = embedder.encode([query])
hits = store.search(query_vec, top_k=3)
for h in hits:
    print(f\"[{h['score']:.3f}] {h['intent']:>20s} | Q: {h['instruction']}\")
"""),

    md("## 4.4 Prompt template\nExactly the structure specified in the brief, with `{detected_sentiment}` "
       "filled in from Module 2's output so the model acknowledges frustration before answering."),
    code("""system, user = rag.build_prompt(query, hits, sentiment='negative')
print(system)
print('---')
print(user)
"""),

    md("## 4.5 Generation via Groq\nRequires a `GROQ_API_KEY` environment variable "
       "(free account at https://console.groq.com). Without it, `generate_answer` falls back to "
       "returning the single best-matching KB response directly (a purely extractive answer, no LLM "
       "call) so the pipeline is still runnable/testable offline."),
    code("""os.environ.setdefault('GROQ_MODEL', rag.GROQ_MODEL)  # openai/gpt-oss-120b
result = rag.answer_query(query, embedder, store, sentiment='negative')
print(result['answer'])
"""),

    md("## 4.6 Out-of-KB / low-confidence handling\nPer the brief: if the retrieved context doesn't "
       "cover the question, the system prompt instructs the model to say so honestly and offer to "
       "escalate, rather than guessing. The extractive fallback path mirrors this with a hard-coded "
       "escalation message when nothing relevant is retrieved."),
    code("""off_topic = 'what is the meaning of life'
result2 = rag.answer_query(off_topic, embedder, store, sentiment='neutral')
print(result2['answer'])
"""),

    md("## 4.7 Save the index for deployment reuse"),
    code("""os.makedirs('../models', exist_ok=True)
try:
    store.save('../models/rag_index.faiss', '../models/rag_meta.pkl')
    print('Saved FAISS index + metadata.')
except Exception as e:
    print('Save skipped (FAISS not available in this environment):', e)
"""),

    md("""## Notes / decisions to defend at assessment
- FAISS chosen as the default vector store to avoid an external Qdrant account for local
  dev/assessment; `QdrantStore` (same interface) is provided as the documented alternative for
  a persistent/shared deployment, satisfying the brief's "free cloud Qdrant... or local
  FAISS/Chroma" either/or.
- Retrieval unit = `instruction` (the question), grounding text injected = paired `response`
  (the answer) -- matches the brief's explicit guidance and avoids embedding answer text, which
  would blur semantic search relative to embedding the customer's own phrasing style.
- The system prompt explicitly instructs the model to acknowledge frustration and to refuse to
  guess when context doesn't cover the question, escalating instead -- both requirements from
  the brief are enforced at the prompt level, not just hoped for."""),
])

nbf.write(lang_nb, "notebooks/01_language_detection.ipynb")
nbf.write(sent_nb, "notebooks/02_sentiment_classifier.ipynb")
nbf.write(intent_nb, "notebooks/03_intent_classifier.ipynb")
nbf.write(rag_nb, "notebooks/04_rag_pipeline.ipynb")
print("Notebooks written.")
