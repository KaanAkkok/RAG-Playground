def reciprocal_rank_fusion(
    sparse_res: list[tuple[str, float]], dense_res: list[tuple[str, float]], k: int = 60
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF) algorithm for combining sparse and dense retrieval results.

    Args:
        sparse_res (list[tuple[str, float]]): A list of tuples containing document IDs and their scores from sparse retrieval (e.g., BM25).
        dense_res (list[tuple[str, float]]): A list of tuples containing document IDs and their scores from dense retrieval (e.g., embeddings).
        k (int): The RRF parameter that controls the influence of lower-ranked documents. Default is 60.

    Returns:
        list[tuple[str, float]]: A list of tuples containing document IDs and their fused scores, sorted by score in descending order.
    """
    sparse_ranks = {
        doc_id: rank for rank, (doc_id, _) in enumerate(sparse_res, start=1)
    }
    dense_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(dense_res, start=1)}
    merged_rankings = {}

    for doc_id, rank in sparse_ranks.items():
        merged_rankings[doc_id] = merged_rankings.get(doc_id, 0.0) + 1.0 / (k + rank)

    for doc_id, rank in dense_ranks.items():
        merged_rankings[doc_id] = merged_rankings.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(merged_rankings.items(), key=lambda item: item[1], reverse=True)
