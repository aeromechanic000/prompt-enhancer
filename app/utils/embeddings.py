"""Embedding utilities for text processing."""

import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity score between -1 and 1.
    """
    a_arr = np.array(a)
    b_arr = np.array(b)

    dot_product = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def normalize_embedding(embedding: list[float]) -> list[float]:
    """Normalize an embedding vector to unit length.

    Args:
        embedding: Input embedding vector.

    Returns:
        Normalized embedding vector.
    """
    arr = np.array(embedding)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return embedding
    return arr / norm


def average_embeddings(embeddings: list[list[float]]) -> list[float]:
    """Compute the average of multiple embeddings.

    Args:
        embeddings: List of embedding vectors.

    Returns:
        Averaged embedding vector.
    """
    if not embeddings:
        return []

    arr = np.array(embeddings)
    return arr.mean(axis=0).tolist()
