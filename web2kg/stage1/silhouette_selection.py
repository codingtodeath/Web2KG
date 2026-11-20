"""Structure-only clustering with an optional near-best, fewest-template rule."""
import math
import time

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances, silhouette_score

from .template_clustering import encode_profiles, kmeans_labels, kmeans_matrix, similarity_matrix
from .page_structure import PAGE_FEATURES
from .templates import structure_fingerprint


def prepare(profiles, *, deduplicate=False, weights=None):
    start = time.perf_counter()
    weights = PAGE_FEATURES if weights is None else weights
    keys = [structure_fingerprint(p, weights) for p in profiles]
    eligible = np.array([not p.get('isolated') and any(
        region.get(f) for region in p['region_features'].values() for f in weights)
        for p in profiles], dtype=bool)
    indices = np.flatnonzero(eligible).tolist()
    if deduplicate:
        first = {}
        for i in indices:
            first.setdefault(keys[i], i)
        # Canonical structural order makes ties independent of arrival order/multiplicity.
        indices = [first[k] for k in sorted(first)]
    positions = {keys[i]: j for j, i in enumerate(indices)}
    inverse = ([positions[keys[i]] for i in np.flatnonzero(eligible)] if deduplicate
               else list(range(len(indices))))
    encoded = encode_profiles([profiles[i] for i in indices], weights)
    x = kmeans_matrix(encoded, weights)
    jac = 1 - similarity_matrix(encoded, weights)
    euclidean = pairwise_distances(x) if x.shape[0] else np.zeros((0, 0))
    # Identical sparse rows can acquire sqrt(machine epsilon) distance through BLAS.
    euclidean[euclidean < 1e-7] = 0
    np.fill_diagonal(jac, 0)
    unique = len(np.unique(x.toarray(), axis=0)) if x.shape[0] else 0
    return dict(eligible=eligible, inverse=np.array(inverse, dtype=int), x=x,
                fingerprints=[keys[i] for i in indices], deduplicated=deduplicate,
                jaccard=jac, euclidean=euclidean, unique_vectors=unique,
                seconds=time.perf_counter()-start)


def candidates(arrays, method, geometry='jaccard', seed=0):
    """Generate each obtainable distance-cut partition, or each feasible K."""
    n = arrays['x'].shape[0]
    if n < 3:
        return []
    rows = []
    if method == 'kmeans':
        for k in range(2, min(n-1, arrays['unique_vectors'])+1):
            labels = kmeans_labels(arrays['x'], np.zeros(n, dtype=bool), k, seed)
            rows.append(dict(parameter=k, labels=labels))
    else:
        if method not in ('average', 'complete'):
            raise ValueError('Unknown method')
        tree = linkage(squareform(arrays[geometry], checks=False), method=method)
        seen = set()
        # t=0 permits the all-unique singleton partition, which is rejected below.
        for height in np.unique(np.r_[0., tree[:, 2]]):
            labels = fcluster(tree, t=float(height), criterion='distance') - 1
            k = len(np.unique(labels))
            if 2 <= k < n and k not in seen:
                seen.add(k)
                rows.append(dict(parameter=float(height), labels=labels))
    return rows


def score_options(arrays, options, geometry='jaccard'):
    n = arrays['x'].shape[0]
    return [dict(item, clusters=len(np.unique(item['labels'])), silhouette=float(
        silhouette_score(arrays[geometry], item['labels'], metric='precomputed')))
        for item in options if 2 <= len(np.unique(item['labels'])) < n]


def select(arrays, options, geometry='jaccard', *, tolerance=0.):
    """Selection takes no annotation, URL, old template ID or class-count input."""
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 2:
        raise ValueError('Silhouette tolerance must be finite and in [0, 2]')
    records = options if all('silhouette' in r for r in options) else score_options(arrays, options, geometry)
    maximum = max((r['silhouette'] for r in records), default=None)
    near = [r for r in records if r['silhouette'] >= maximum - tolerance - 1e-12]
    best = min(near, key=lambda r: (r['clusters'], -r['silhouette'], r['parameter'])) if near else None
    # Undefined silhouette must not merge unrelated structures (0/1/2 points or tied tree).
    unique = {key: i for i, key in enumerate(sorted(set(arrays['fingerprints'])))}
    labels = best['labels'] if best else np.array([unique[k] for k in arrays['fingerprints']], dtype=int)
    full = np.empty(len(arrays['eligible']), dtype=int)
    full[arrays['eligible']] = labels[arrays['inverse']]
    next_label = int(max(labels, default=-1))+1
    for index in np.flatnonzero(~arrays['eligible']):
        full[index] = next_label
        next_label += 1
    return dict(status='selected' if best else 'unidentifiable',
                parameter=best['parameter'] if best else None,
                silhouette=best['silhouette'] if best else None,
                maximum_silhouette=maximum, tolerance=tolerance,
                selection_points=arrays['x'].shape[0], deduplicated=arrays['deduplicated'],
                clusters=len(np.unique(full)), predicted=full.tolist(),
                candidates=[{k: v for k, v in r.items() if k != 'labels'} for r in records])
