import numpy as np
import faiss

def rbf_cka(embeddings, ids, n):
    ids = np.array(ids)
    embeddings = np.array(embeddings)
    original_embeddings = []
    augmented_embeddings= []
    uids = set(ids)
    for id in uids:
       id_original_index = np.where(ids==id)[0][0]
       id_augmented_indices = np.where(ids==id)[0][1:]

       id_original_embeddings = [embeddings[id_original_index]]*n
       id_augmented_embeddings = embeddings[id_augmented_indices]

       assert len(id_original_embeddings) == len(id_augmented_embeddings)

       original_embeddings.append(id_original_embeddings)
       augmented_embeddings.append(id_augmented_embeddings)

    return _rbf_cka(original_embeddings, augmented_embeddings, True)

def linear_cka(embeddings, ids, n):
    ids = np.array(ids)
    embeddings = np.array(embeddings)
    original_embeddings = []
    augmented_embeddings= []
    uids = set(ids)
    for id in uids:
       id_original_index = np.where(ids==id)[0][0]
       id_augmented_indices = np.where(ids==id)[0][1:]

       id_original_embeddings = [embeddings[id_original_index]]*n
       id_augmented_embeddings = embeddings[id_augmented_indices]

       assert len(id_original_embeddings) == len(id_augmented_embeddings)

       original_embeddings.append(id_original_embeddings)
       augmented_embeddings.append(id_augmented_embeddings)

    return _linear_cka(original_embeddings, augmented_embeddings, True)


def top_k_augmentations_recall(embeddings, ids, k, n):
    """
    ids: list of original and augmented images ids .e.g. 111111222222...
    """
    ids = np.array(ids)
    embeddings = np.array(embeddings)
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
    embeddings = np.array(embeddings)
    # Generate index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    avg_ranks, min_ranks, max_ranks = [], [], []
    uids = set(ids)
    for id in uids:
        original_idx= np.where(ids==id)[0][0]
        transformed_idx = np.where(ids==id)[0][1:]
        _, neighbors = index.search(embeddings[original_idx:original_idx + 1], len(embeddings))
        neighbors = neighbors[0][1:] # skip self
        ranks = [np.where(neighbors == t)[0][0].item() + 1 for t in transformed_idx] # +1 to rank from 1
        avg_ranks.append(np.mean(ranks).item())
        min_ranks.append(np.min(ranks).item())
        max_ranks.append(np.max(ranks).item())

    return avg_ranks, min_ranks, max_ranks

def _gram_linear(x):
  return x.dot(x.T)

def _gram_rbf(x, threshold=1.0):
  dot_products = x.dot(x.T)
  sq_norms = np.diag(dot_products)
  sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
  sq_median_distance = np.median(sq_distances)
  return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))

def _center_gram(gram, unbiased=False):
  if not np.allclose(gram, gram.T):
    raise ValueError('Input must be a symmetric matrix.')
  gram = gram.copy()
  if unbiased:
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
  else:
    means = np.mean(gram, 0, dtype=np.float64)
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

  return gram

def _cka(gram_x, gram_y, debiased=False):
  gram_x = _center_gram(gram_x, unbiased=debiased)
  gram_y = _center_gram(gram_y, unbiased=debiased)
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel())
  normalization_x = np.linalg.norm(gram_x)
  normalization_y = np.linalg.norm(gram_y)
  return scaled_hsic / (normalization_x * normalization_y)

def _rbf_cka(x, y, debiased=True):
   gram_x = _gram_rbf(x)
   gram_y = _gram_rbf(y)
   return _cka(gram_x, gram_y, debiased)

def _linear_cka(x, y, debiased=True):
    gram_x = _gram_linear(x)
    gram_y = _gram_linear(y)
    return _cka(gram_x, gram_y, debiased)  

def _test_metrics():
    embeddings = np.random.random((20, 512))
    embeddings = [[embeddings[i,:]]*5 for i in range(0, 20)]
    embeddings = [e for e5 in embeddings for e in e5]
    embeddings = np.array(embeddings)
    assert embeddings.shape == (100, 512)
    
    ids = [ [i]*5 for i in range(1,21)]
    ids = [j for i in ids for j in i]
    
    recalls = top_k_augmentations_recall(embeddings, ids, 4, 4)
    assert recalls == [1]*20
    
    avg_ranks, min_ranks, max_ranks = augmentations_rank(embeddings, ids)
    assert avg_ranks == [np.mean([1,2,3,4])]*20
    assert min_ranks == [1]*20
    assert max_ranks == [4]*20

    rbf_cka_value = rbf_cka(embeddings, ids, 4)
    linear_cka_value = linear_cka(embeddings, ids, 4)

    print(f"RBF_CKA: {rbf_cka_value}")
    print(f"Linear_CKA: {linear_cka_value}")

