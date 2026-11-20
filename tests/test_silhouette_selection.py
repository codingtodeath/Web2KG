import numpy as np
import pytest

from web2kg.stage1.silhouette_selection import candidates, prepare, select


def profile(*features, isolated=False):
    return {'region_features': {'main': {'structure': list(features), 'combinations': []}}, 'isolated': isolated}


@pytest.mark.parametrize('method', ['average', 'complete', 'kmeans'])
def test_two_groups_selected_without_labels(method):
    if method == 'kmeans':
        pytest.importorskip('threadpoolctl', minversion='3.6.0')
    a = prepare([profile('a')]*3 + [profile('b')]*3)
    found = select(a, candidates(a, method))
    assert found['clusters'] == 2
    assert found['silhouette'] == pytest.approx(1)
    assert found['predicted'][:3] == [found['predicted'][0]]*3
    assert found['predicted'][0] != found['predicted'][-1]


def test_singletons_and_all_identical_cannot_fake_silhouette():
    a = prepare([profile('a')]*5)
    assert select(a, candidates(a, 'average'))['status'] == 'unidentifiable'
    assert select(a, [{'parameter': 5, 'labels': np.arange(5)}])['silhouette'] is None


def test_isolates_excluded_and_remain_separate():
    a = prepare([profile('a')]*3 + [profile('b')]*3 + [profile('a', isolated=True), profile()])
    result = select(a, candidates(a, 'complete'))
    assert result['silhouette'] == pytest.approx(1)
    assert result['clusters'] == 4
    assert len(set(result['predicted'][-2:])) == 2


def test_tied_tree_heights_only_search_realizable_cuts():
    a = prepare([profile('a')]*2 + [profile('b')]*2 + [profile('c')]*2)
    opts = candidates(a, 'average')
    assert [len(set(r['labels'])) for r in opts] == [3]


def test_no_eligible_pages_reports_unidentifiable():
    a = prepare([profile(), profile()])
    result = select(a, candidates(a, 'average'))
    assert result['status'] == 'unidentifiable'
    assert result['predicted'] == [0, 1]


def test_deduplication_restores_every_page_and_ignores_multiplicity():
    profiles = [profile('a'), profile('a', 'x'), profile('b'), profile('b', 'y')]
    def run(rows):
        a = prepare(rows, deduplicate=True)
        return select(a, candidates(a, 'average'), tolerance=.05)
    unique = run(profiles)
    repeated = run([profiles[0]]*20 + profiles[1:])
    assert repeated['selection_points'] == 4
    assert repeated['predicted'][20:] == unique['predicted'][1:]
    assert repeated['predicted'][:20] == [unique['predicted'][0]]*20
    assert repeated['silhouette'] == pytest.approx(unique['silhouette'])
    reverse = run(list(reversed(profiles)))
    assert reverse['predicted'][::-1] == unique['predicted']


def test_near_best_uses_smallest_cluster_count_not_the_highest_score():
    a = prepare([profile(str(i)) for i in range(6)])
    options = [dict(parameter=.1, labels=np.array([0, 0, 1, 1, 2, 2]), clusters=3, silhouette=.70),
               dict(parameter=.2, labels=np.array([0, 0, 0, 1, 1, 1]), clusters=2, silhouette=.65)]
    assert select(a, options, tolerance=.049)['clusters'] == 3
    assert select(a, options, tolerance=.05)['clusters'] == 2
    with pytest.raises(ValueError):
        select(a, options, tolerance=float('nan'))


@pytest.mark.parametrize('profiles,expected', [([], 0), ([profile('a')]*3, 1),
    ([profile('a'), profile('b')], 2), ([profile('a'), profile('b'), profile('c')], 3),
    ([profile('a'), profile('a', isolated=True), profile('a', isolated=True)], 3)])
def test_undefined_silhouette_only_combines_exact_duplicates(profiles, expected):
    a = prepare(profiles, deduplicate=True)
    result = select(a, candidates(a, 'average'))
    assert result['clusters'] == expected
    assert result['silhouette'] is None
