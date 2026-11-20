"""Sparse structural encoding and clustering shared by experiments and batched maintenance."""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from .page_structure import PAGE_FEATURES


@dataclass
class Encoding:
    blocks: dict
    regions: dict
    isolated: np.ndarray
    size: int

    @property
    def dimensions(self):
        return sum(x.shape[1] for x in self.blocks.values())


def encode_profiles(profiles, weights=None):
    """Binary CSR blocks; numeric int64 multiplication counts shared features."""
    weights = PAGE_FEATURES if weights is None else weights
    n = len(profiles)
    regions = sorted({r for p in profiles for r in p['region_features']})
    present = {r: np.array([r in p['region_features'] for p in profiles]) for r in regions}
    blocks = {}
    for region in regions:
        for feature in weights:
            sets = [set(p['region_features'].get(region, {}).get(feature, [])) for p in profiles]
            vocabulary = {v: i for i, v in enumerate(sorted(set().union(*sets)))}
            rows, cols = [], []
            for row, values in enumerate(sets):
                for value in sorted(values):
                    rows.append(row)
                    cols.append(vocabulary[value])
            blocks[region, feature] = sparse.csr_matrix(
                (np.ones(len(rows), dtype=np.int64), (rows, cols)), shape=(n, len(vocabulary)))
    empty = np.ones(n, dtype=bool)
    for block in blocks.values():
        empty &= np.asarray(block.sum(axis=1)).ravel() == 0
    isolated = empty | np.array([bool(p.get('isolated')) for p in profiles], dtype=bool)
    return Encoding(blocks, present, isolated, n)


def similarity_matrix(encoded, weights=None):
    """Exactly preserve page_similarity's pair-dependent missing-region weighting."""
    weights = PAGE_FEATURES if weights is None else weights
    shape = (encoded.size, encoded.size)
    main_sum, aux_sum, main_count, aux_count = [np.zeros(shape) for _ in range(4)]
    for region, presence in encoded.regions.items():
        numerator, denominator = np.zeros(shape), np.zeros(shape)
        for feature, weight in weights.items():
            x = encoded.blocks[region, feature]
            counts = np.asarray(x.sum(axis=1)).ravel()
            intersection = (x @ x.T).toarray()
            union = counts[:, None] + counts[None, :] - intersection
            valid = union > 0
            numerator += weight * np.divide(intersection, union, out=np.zeros(shape), where=valid)
            denominator += weight * valid
        score = np.divide(numerator, denominator, out=np.zeros(shape), where=denominator > 0)
        present = presence[:, None] | presence[None, :]
        if region.split('/')[-1] == 'main':
            main_sum += score
            main_count += present
        else:
            aux_sum += score
            aux_count += present
    main = np.divide(main_sum, main_count, out=np.zeros(shape), where=main_count > 0)
    auxiliary = np.divide(aux_sum, aux_count, out=np.zeros(shape), where=aux_count > 0)
    both = (main_count > 0) & (aux_count > 0)
    result = np.where(both, .85 * main + .15 * auxiliary, main + auxiliary)
    result[encoded.isolated, :] = 0
    result[:, encoded.isolated] = 0
    return np.clip(result, 0, 1)


def kmeans_matrix(encoded, weights=None):
    """Same feature vocabulary, native Euclidean geometry with block normalization.

    Each nonempty block has unit L2 norm before square-root region/feature weighting.
    This prevents large regions from dominating solely through feature counts;
    it does NOT claim equivalence to the pair-dependent Jaccard metric.
    """
    weights = PAGE_FEATURES if weights is None else weights
    main = [r for r in encoded.regions if r.split('/')[-1] == 'main']
    aux = [r for r in encoded.regions if r not in main]
    n = encoded.size
    mc = sum((encoded.regions[r].astype(float) for r in main), np.zeros(n))
    ac = sum((encoded.regions[r].astype(float) for r in aux), np.zeros(n))
    both = (mc > 0) & (ac > 0)
    blocks = []
    for region in encoded.regions:
        count = mc if region in main else ac
        allocation = np.where(both, .85 if region in main else .15, 1.)
        region_weight = np.divide(allocation, count, out=np.zeros(n), where=count > 0)
        feature_sum = sum((weight * (np.asarray(encoded.blocks[region, f].sum(axis=1)).ravel() > 0)
                           for f, weight in weights.items()), np.zeros(n))
        for feature, weight in weights.items():
            x = encoded.blocks[region, feature]
            if not x.shape[1]:
                continue
            scales = np.sqrt(np.divide(region_weight * weight, feature_sum,
                                      out=np.zeros(n), where=feature_sum > 0))
            blocks.append(normalize(x.astype(float), norm='l2').multiply(scales[:, None]).tocsr())
    return sparse.hstack(blocks, format='csr') if blocks else sparse.csr_matrix((n, 0))


def _with_isolates(size, eligible, labels):
    result = np.full(size, -1, dtype=int)
    result[eligible] = labels
    next_label = int(max(labels, default=-1)) + 1
    for i in np.flatnonzero(~eligible):
        result[i] = next_label
        next_label += 1
    return result


def hierarchical_labels(similarity, isolated, method='average', threshold=.85):
    if method not in {'average', 'complete'}:
        raise ValueError('Expected average or complete linkage')
    eligible = ~np.asarray(isolated, dtype=bool)
    n = int(eligible.sum())
    labels = np.zeros(n, dtype=int)
    if n > 1:
        distances = 1 - similarity[np.ix_(eligible, eligible)]
        np.fill_diagonal(distances, 0)
        tree = linkage(squareform(distances, checks=False), method=method)
        labels = fcluster(tree, t=1 - threshold, criterion='distance') - 1
    return _with_isolates(len(isolated), eligible, labels)


def kmeans_labels(matrix, isolated, clusters, seed=0):
    eligible = ~np.asarray(isolated, dtype=bool)
    if clusters < 1 or clusters > eligible.sum():
        raise ValueError('K must fit the number of non-isolated pages')
    labels = KMeans(n_clusters=clusters, n_init=20, max_iter=300,
                    algorithm='lloyd', random_state=seed).fit_predict(matrix[eligible])
    return _with_isolates(len(isolated), eligible, labels)
