import numpy as np
import faiss

def top_k_augmentations_recall(embeddings, ids, k, n):
    """
    ids: list of original and augmented images ids .e.g. 111111222222...
    """
    ids = np.array(ids)
    # Generate index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    recalls = []
    uids = set(ids)
    for id in uids:
        original_index = np.where(ids==id)[0][0] # The first is always the original
        _, neighbors = index.search(embeddings[original_index:original_index+1], k + 1)  # +1 for self
        neighbors = neighbors[0][1:]  # skip self
        hits = sum(ids[n] == id for n in neighbors)
        recalls.append((hits / n).item())
    return list(recalls)


def augmentations_rank(embeddings, ids):
    ids = np.array(ids)
    # Generate index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    avg_ranks, min_ranks, max_ranks = [], [], []
    uids = set(ids)
    for id in uids:
        original_idx= np.where(ids==id)[0][0]
        transformed_idx = np.where(ids==id)[0][1:]
        _, neighbors = index.search(embeddings[original_idx:original_idx+1], len(embeddings))
        neighbors = neighbors[0][1:] # skip self
        ranks = [np.where(neighbors == t)[0][0].item() for t in transformed_idx]
        avg_ranks.append(np.mean(ranks).item())
        min_ranks.append(np.min(ranks).item())
        max_ranks.append(np.max(ranks).item())

    return avg_ranks, min_ranks, max_ranks

def _test_metrics():
    embeddings = np.random.random((20, 512))
    embeddings = [[embeddings[i,:]]*5 for i in range(0, 20)]
    embeddings = [e for e5 in embeddings for e in e5]
    embeddings = np.array(embeddings)
    assert embeddings.shape == (100, 512)
    ids = [ [i]*5 for i in range(1,21)]
    ids = [j for i in ids for j in i]
    recalls = top_k_augmentations_recall(embeddings, ids, 5, 5)
    assert recalls == [1]*20