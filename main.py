"""
FastAPI deployment for the RAG-based e-commerce customer support chatbot.

Run locally with:
    uvicorn app.main:app --reload --port 8000

On startup this either loads previously-trained models from `models/` (if
present -- run `python -m scripts.train_all` first) or trains fresh ones on
the fly using the bundled sample data / real HF datasets (whichever is
available), so the app is runnable out of the box either way.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel

from src import intent_classifier, language_detection, rag_pipeline, sentiment_classifier
from src.pipeline import SupportPipeline

app = FastAPI(
    title="E-commerce Support Chatbot API",
    description="RAG-based customer support chatbot: language -> sentiment -> intent -> route -> respond.",
    version="1.0.0",
)

_pipeline: SupportPipeline | None = None


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    language: dict
    sentiment: dict
    intent: dict
    routing_behavior: str
    retrieved_context: list


def _load_or_train() -> SupportPipeline:
    # Language detector
    try:
        lang_pipe = language_detection.load()
    except Exception:
        train_df, val_df, _ = language_detection.load_hf_dataset()
        lang_pipe = language_detection.train(train_df, val_df)
        os.makedirs("models", exist_ok=True)
        language_detection.save(lang_pipe)

    # Sentiment classifier
    try:
        sentiment_bundle = sentiment_classifier.load()
    except Exception:
        sdf = sentiment_classifier.load_hf_dataset()
        sentiment_bundle = sentiment_classifier.train(sdf)
        sentiment_classifier.save(*sentiment_bundle)

    # Intent classifier
    try:
        intent_pipe = intent_classifier.load()
    except Exception:
        idf = intent_classifier.load_hf_dataset()
        intent_pipe, _, _ = intent_classifier.train(idf)
        intent_classifier.save(intent_pipe)

    # RAG store (rebuilt each startup; swap for FaissStore.load() once saved)
    rdf = rag_pipeline.load_hf_dataset()
    embedder = rag_pipeline.Embedder()
    store = rag_pipeline.build_index(rdf, embedder)

    return SupportPipeline(lang_pipe, sentiment_bundle, intent_pipe, embedder, store)


@app.on_event("startup")
def startup_event():
    global _pipeline
    print("Loading/training pipeline models... this may take a moment.")
    _pipeline = _load_or_train()
    print("Pipeline ready.")


@app.get("/health")
def health():
    return {"status": "ok", "pipeline_loaded": _pipeline is not None}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if _pipeline is None:
        return ChatResponse(
            answer="The service is still starting up, please try again in a moment.",
            language={}, sentiment={}, intent={}, routing_behavior="unavailable",
            retrieved_context=[],
        )
    result = _pipeline.handle_message(req.message)
    return ChatResponse(**result)
