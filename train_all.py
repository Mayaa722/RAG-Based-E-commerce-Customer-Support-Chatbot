"""
Train and persist all four models to `models/` so the FastAPI app can start
quickly by loading them instead of retraining on every launch.

Usage:
    python -m scripts.train_all
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import intent_classifier, language_detection, rag_pipeline, sentiment_classifier


def main():
    os.makedirs("models", exist_ok=True)

    print("\n=== 1) Language Detection ===")
    train_df, val_df, test_df = language_detection.load_hf_dataset()
    lang_pipe = language_detection.train(train_df, val_df)
    language_detection.evaluate(lang_pipe, test_df)
    language_detection.save(lang_pipe)

    print("\n=== 2) Sentiment / Emotion Classifier ===")
    sdf = sentiment_classifier.load_hf_dataset()
    sentiment_bundle = sentiment_classifier.train(sdf)
    sentiment_classifier.save(*sentiment_bundle)

    print("\n=== 3) Intent Classifier ===")
    idf = intent_classifier.load_hf_dataset()
    intent_pipe, _, _ = intent_classifier.train(idf)
    intent_classifier.save(intent_pipe)

    print("\n=== 4) RAG Index (built at app startup, but sanity-checked here) ===")
    rdf = rag_pipeline.load_hf_dataset()
    embedder = rag_pipeline.Embedder()
    store = rag_pipeline.build_index(rdf, embedder)
    print(f"Indexed {len(store.meta)} support KB entries.")

    print("\nAll models trained and saved to models/.")


if __name__ == "__main__":
    main()
