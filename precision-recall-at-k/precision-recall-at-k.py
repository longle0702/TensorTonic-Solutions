def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    top_k = recommended[:k]
    hits = []
    for i in top_k:
        if i in relevant:
            hits.append(i)
    pre = len(hits) / k
    rec = len(hits) / len(relevant)
    return [pre, rec]
    