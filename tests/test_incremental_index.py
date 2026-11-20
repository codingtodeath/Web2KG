import copy
import json
import shutil

import numpy as np
import pytest
from PIL import Image

from test_visual_navigation import FakeJina
from web2kg.common.jina import sha256
from web2kg.common.storage import EventLog, Graph
from web2kg.stage1.incremental_index import extend_visual_index
from web2kg.stage1.visual_index import build_visual_index
from web2kg.stage2.retrieval import retrieve


def add_page(graph, key):
    graph.add_node({'id': key, 'title': key, 'url': 'http://test/' + key,
                    'headings': [], 'actions': [], 'text': ''}, [])
    if graph.data['root'] is None:
        graph.data.update(root=key, start_url='http://test/a')
    else:
        graph.add_edge(graph.data['root'], key, {'id': key, 'kind': 'click', 'role': 'link',
            'name': key, 'href': 'http://test/' + key})
    path = graph.directory / 'snapshots' / key / 'page.png'
    path.parent.mkdir(parents=True)
    Image.new('RGB', (20, 80), 'white').save(path)
    graph.save()


def setup_graphs(tmp_path, tiled=False, fresh=True):
    base, client = Graph(tmp_path / 'base'), FakeJina()
    add_page(base, 'a')
    index = build_visual_index(base, client, tiles=tiled, tile_height=40, overlap=8)
    task = Graph(tmp_path / 'task')
    task.data = copy.deepcopy(base.data)
    shutil.copytree(base.directory / 'snapshots', task.directory / 'snapshots')
    if fresh:
        add_page(task, 'b')
    client.calls.clear()
    return base, task, client, index


@pytest.mark.parametrize('tiled', [False, True])
def test_extension_reuses_vectors_and_retrieves_new_page(tmp_path, tiled):
    base, task, client, source = setup_graphs(tmp_path, tiled)
    original = {p.relative_to(base.directory): sha256(p) for p in base.directory.rglob('*') if p.is_file()}
    index = extend_visual_index(base, task, client, EventLog(task.directory))
    assert index['incremental']['reused_ids'] == ['a']
    assert index['incremental']['encoded_ids'] == ['b']
    assert len(client.calls) == (4 if tiled else 1)
    assert all(value.parent.name == 'b' or value.name.startswith('b-') for value, _ in client.calls)
    with np.load(base.directory / source['vectors']) as old, np.load(task.directory / index['vectors']) as new:
        assert all(np.array_equal(old[k], new[k]) for k in old.files)
    if tiled:
        assert index['tiling']['height'] == 40 and len(index['tiling']['pages']['b']) == 3
        for rows in index['tiling']['pages'].values():
            assert all(sha256(task.directory / row['image']) == row['sha256'] for row in rows)
    assert retrieve(task, 'find the new page', client, 1)[0]['id'] == 'b'
    assert {p.relative_to(base.directory): sha256(p) for p in base.directory.rglob('*') if p.is_file()} == original
    assert json.loads((task.directory / 'index.json').read_text()) == index


@pytest.mark.parametrize('changed', ['encoder', 'source_image', 'task_image', 'vectors'])
def test_extension_rejects_changed_provenance_before_encoding(tmp_path, changed):
    base, task, client, source = setup_graphs(tmp_path)
    if changed == 'encoder':
        client.version = 'different'
    elif changed == 'vectors':
        (base.directory / source['vectors']).write_bytes(b'changed')
    else:
        owner = base if changed == 'source_image' else task
        (owner.directory / 'snapshots/a/page.png').write_bytes(b'changed')
    with pytest.raises(ValueError):
        extend_visual_index(base, task, client, EventLog(task.directory))
    assert not client.calls and not (task.directory / 'index.json').exists()


def test_encoding_failure_does_not_publish_partial_index(tmp_path):
    from unittest.mock import Mock
    base, task, client, _ = setup_graphs(tmp_path)
    client.encode = Mock(side_effect=ConnectionError('local encoder stopped'))
    with pytest.raises(ConnectionError):
        extend_visual_index(base, task, client, EventLog(task.directory))
    assert client.encode.call_count == 1 and not (task.directory / 'index.json').exists()


def test_no_new_page_needs_no_encoding(tmp_path):
    base, task, client, _ = setup_graphs(tmp_path, fresh=False)
    index = extend_visual_index(base, task, client, EventLog(task.directory))
    assert not client.calls and index['incremental']['encoded_ids'] == []
    assert index['ids'] == ['a']


@pytest.mark.parametrize('tiled', [False, True])
def test_removed_source_leaf_is_not_encoded_or_required_in_task(tmp_path, tiled):
    base, task, client, _ = setup_graphs(tmp_path, tiled, fresh=False)
    add_page(base, 'retired')
    source = build_visual_index(base, client, tiles=tiled, tile_height=40, overlap=8)
    client.calls.clear()
    index = extend_visual_index(base, task, client, EventLog(task.directory))
    assert index['ids'] == ['a'] and index['incremental']['removed_ids'] == ['retired']
    assert not client.calls
    assert set(index['images_sha256']) == {'a'}
    if tiled:
        assert set(index['tiling']['pages']) == {'a'}
    with np.load(base.directory / source['vectors']) as old, np.load(task.directory / index['vectors']) as new:
        assert not any(k.startswith('retired') for k in new.files)
        assert all(np.array_equal(new[k], old[k]) for k in new.files)
    assert retrieve(task, 'page', client, 5)[0]['id'] == 'a'
