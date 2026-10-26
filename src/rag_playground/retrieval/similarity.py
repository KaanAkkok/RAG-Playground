def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate the cosine similarity between two vectors.

    Args:
        vec1 (list[float]): First vector.
        vec2 (list[float]): Second vector.

    Returns:
        float: Cosine similarity between vec1 and vec2. (btw 0 and 1)
        indicates near to 1 means the vectors are similar, and near to 0 means they are dissimilar.
    """
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_vec1 = sum(a * a for a in vec1) ** 0.5
    norm_vec2 = sum(b * b for b in vec2) ** 0.5
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)
