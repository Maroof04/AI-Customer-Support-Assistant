"""
Knowledge Base and FAISS Vector Retriever Module.
Encodes customer support knowledge articles into dense semantic vectors
and performs fast similarity-based retrieval using FAISS IndexFlatIP.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import faiss

# Check for SentenceTransformers availability
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class DenseSemanticEncoder:
    """
    Fallback dense semantic vector encoder using TF-IDF + Latent Semantic Analysis (TruncatedSVD)
    when heavyweight PyTorch/SentenceTransformers environments are not loaded.
    Produces unit-normalized dense float32 vectors.
    """

    def __init__(self, n_components: int = 64):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode"
        )
        self.svd: Optional[TruncatedSVD] = None
        self.is_fitted = False

    def fit(self, texts: List[str]) -> "DenseSemanticEncoder":
        from sklearn.decomposition import TruncatedSVD

        tfidf_mat = self.vectorizer.fit_transform(texts)
        actual_components = min(self.n_components, tfidf_mat.shape[1] - 1, tfidf_mat.shape[0] - 1)
        actual_components = max(8, actual_components)
        self.svd = TruncatedSVD(n_components=actual_components, random_state=42)
        self.svd.fit(tfidf_mat)
        self.is_fitted = True
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted or self.svd is None:
            raise ValueError("Encoder must be fitted first.")
        tfidf_mat = self.vectorizer.transform(texts)
        dense = self.svd.transform(tfidf_mat).astype(np.float32)
        # Normalize each vector to unit length (L2 norm)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized = dense / norms
        return normalized.astype(np.float32)


class FaissRetriever:
    """
    Retrieves the most relevant FAQ articles from the structured knowledge base
    using FAISS cosine similarity index (IndexFlatIP with normalized vectors).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.neural_encoder = None
        self.fallback_encoder: Optional[DenseSemanticEncoder] = None
        self.use_neural: bool = False
        self.dimension: Optional[int] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.kb_df: Optional[pd.DataFrame] = None
        self.articles_metadata: List[Dict[str, Any]] = []

        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.neural_encoder = SentenceTransformer(self.model_name)
                self.use_neural = True
            except Exception:
                self.use_neural = False

    def _prepare_document_text(self, row: pd.Series) -> str:
        """
        Combines structured fields into a dense textual representation for indexing.
        """
        category = row.get("category", "")
        title = row.get("title", "")
        question = row.get("question", "")
        keywords = row.get("keywords", "")
        answer = row.get("answer", "")
        return f"Category: {category}. Title: {title}. Question: {question}. Keywords: {keywords}. Answer: {answer}"

    def fit_from_dataframe(self, df: pd.DataFrame) -> "FaissRetriever":
        """
        Builds the FAISS index from knowledge_base DataFrame.
        """
        self.kb_df = df.copy()
        doc_texts = [self._prepare_document_text(row) for _, row in df.iterrows()]
        self.articles_metadata = df.to_dict(orient="records")

        if self.use_neural and self.neural_encoder is not None:
            embeddings = self.neural_encoder.encode(doc_texts, show_progress_bar=False)
            embeddings = np.array(embeddings, dtype=np.float32)
            # L2 normalize embeddings for cosine similarity with IndexFlatIP
            faiss.normalize_L2(embeddings)
        else:
            self.fallback_encoder = DenseSemanticEncoder(n_components=64)
            self.fallback_encoder.fit(doc_texts)
            embeddings = self.fallback_encoder.encode(doc_texts)

        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        return self

    def _encode_query(self, query: str) -> np.ndarray:
        """
        Encodes a single query text into an L2-normalized float32 vector of shape (1, dimension).
        """
        if self.use_neural and self.neural_encoder is not None:
            vec = self.neural_encoder.encode([query], show_progress_bar=False)
            vec = np.array(vec, dtype=np.float32)
            faiss.normalize_L2(vec)
            return vec
        elif self.fallback_encoder is not None:
            vec = self.fallback_encoder.encode([query])
            return vec
        else:
            raise ValueError("Retriever has not been fitted or loaded.")

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.20,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches the knowledge base for top-k articles closest in semantic similarity to the query.
        Returns article details with similarity scores.
        """
        if self.index is None or not self.articles_metadata:
            raise ValueError("Retriever index is empty. Call fit_from_dataframe() or load() first.")

        query_vec = self._encode_query(query)
        # Search for slightly more candidates if category filtering is active
        fetch_k = min(len(self.articles_metadata), top_k * 2 if category_filter else top_k)

        scores, indices = self.index.search(query_vec, fetch_k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.articles_metadata):
                continue
            sim_score = float(score)
            article = self.articles_metadata[idx]

            # Optional intent category filtering
            if category_filter and article.get("category") != category_filter:
                # Still include if high semantic match, or prioritize matched category
                pass

            article_res = dict(article)
            article_res["similarity_score"] = round(sim_score, 4)
            article_res["is_relevant"] = sim_score >= threshold
            results.append(article_res)

        # Sort by similarity score descending
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]

    def save(self, directory: str) -> None:
        """
        Saves the FAISS index and metadata to the specified directory.
        """
        os.makedirs(directory, exist_ok=True)
        index_file = os.path.join(directory, "faiss_index.bin")
        meta_file = os.path.join(directory, "retriever_metadata.joblib")

        faiss.write_index(self.index, index_file)
        payload = {
            "articles_metadata": self.articles_metadata,
            "dimension": self.dimension,
            "use_neural": self.use_neural,
            "model_name": self.model_name,
            "fallback_encoder": self.fallback_encoder
        }
        joblib.dump(payload, meta_file)

    def load(self, directory: str) -> "FaissRetriever":
        """
        Loads the FAISS index and article metadata from disk.
        """
        index_file = os.path.join(directory, "faiss_index.bin")
        meta_file = os.path.join(directory, "retriever_metadata.joblib")

        if not os.path.exists(index_file) or not os.path.exists(meta_file):
            raise FileNotFoundError(f"Retriever files not found in: {directory}")

        self.index = faiss.read_index(index_file)
        payload = joblib.load(meta_file)

        self.articles_metadata = payload["articles_metadata"]
        self.dimension = payload["dimension"]
        self.use_neural = payload["use_neural"]
        self.model_name = payload["model_name"]
        self.fallback_encoder = payload.get("fallback_encoder")

        if self.use_neural and HAS_SENTENCE_TRANSFORMERS and self.neural_encoder is None:
            try:
                self.neural_encoder = SentenceTransformer(self.model_name)
            except Exception:
                pass

        return self
