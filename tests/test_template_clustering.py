import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from web2kg.stage1.page_structure import page_similarity
from web2kg.stage1.template_clustering import (encode_profiles, similarity_matrix,
    kmeans_matrix, hierarchical_labels, kmeans_labels)


def profile(regions, isolated=False):
    return {'region_features': regions, 'evidence': {'sufficient': True}, 'isolated': isolated}


@pytest.mark.parametrize('weights', [None, {'structure': .4, 'combinations': .4, 'controls': .2}])
def test_matrix_preserves_pair_specific_missing_region_and_feature_weights(weights):
    rng = np.random.default_rng(123)
    pages = []
    for _ in range(40):
        regions = {}
        for region in ['main', 'dialog/main', 'navigation', 'body']:
            if rng.random() < .65:
                regions[region] = {f: [str(i) for i in range(10) if rng.random() < .4]
                                   for f in ['structure', 'combinations', 'controls']}
        pages.append(profile(regions or {'body': {'structure': ['fallback']}}))
    actual = similarity_matrix(encode_profiles(pages, weights), weights)
    expected = np.array([[page_similarity(a, b, weights) for b in pages] for a in pages])
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_numeric_product_does_not_overflow_or_use_boolean_multiplication():
    pages = [profile({'main': {'structure': [str(i) for i in range(n)]}}) for n in [512, 1024]]
    np.testing.assert_allclose(similarity_matrix(encode_profiles(pages)), [[1., .5], [.5, 1.]])


@pytest.mark.parametrize('method', ['average', 'complete', 'kmeans'])
def test_empty_or_explicitly_isolated_pages_stay_singletons(method):
    pages = [profile({'main': {'structure': ['x']}}) for _ in range(2)]
    pages += [profile({}), profile({}), profile({'main': {'structure': ['x']}}, True)]
    encoded = encode_profiles(pages)
    if method == 'kmeans':
        pytest.importorskip('threadpoolctl', minversion='3.6.0', reason='Use the documented isolated clustering runtime')
        with threadpool_limits(limits=1):
            labels = kmeans_labels(kmeans_matrix(encoded), encoded.isolated, 1)
    else:
        labels = hierarchical_labels(similarity_matrix(encoded), encoded.isolated, method)
    assert labels[0] == labels[1]
    assert len(set(labels)) == 4


def test_average_linkage_is_member_weighted_and_differs_from_complete():
    similarity = np.array([[1, .9, .82], [.9, 1, .88], [.82, .88, 1]])
    # The last merge has average similarity .85, minimum .82.
    assert len(set(hierarchical_labels(similarity, [False] * 3, 'average', .849))) == 1
    assert len(set(hierarchical_labels(similarity, [False] * 3, 'complete', .849))) == 2


def test_kmeans_uses_content_features_not_row_positions():
    pytest.importorskip('threadpoolctl', minversion='3.6.0', reason='Use the documented isolated clustering runtime')
    pages = [profile({'main': {'structure': [f'{family}/a', f'{family}/b']}})
             for family in ['one', 'two', 'three'] for _ in range(4)]
    encoded = encode_profiles(pages)
    x = kmeans_matrix(encoded)
    order = np.random.default_rng(42).permutation(len(pages))
    with threadpool_limits(limits=1):
        first = kmeans_labels(x, encoded.isolated, 3)
        shuffled = kmeans_labels(x[order], encoded.isolated[order], 3)
    restored = np.empty_like(shuffled)
    restored[order] = shuffled
    assert adjusted_rand_score(first, restored) == 1
    assert adjusted_rand_score(first, np.repeat([0, 1, 2], 4)) == 1
