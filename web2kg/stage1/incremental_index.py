"""Extend a task copy's screenshot index without re-encoding historical pages."""
import copy
import json
import shutil
from uuid import uuid4

import numpy as np
from PIL import Image

from ..common.jina import normalize, sha256
from ..common.page_text import fingerprint, page_text
from ..common.storage import write_json
from .visual_index import tile_boxes


def checked_file(directory, relative, expected):
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()) or sha256(path) != expected:
        raise ValueError('Source index file changed or escaped its graph directory')
    return path


def extend_visual_index(base, graph, client, log):
    if base.directory.resolve() == graph.directory.resolve():
        raise ValueError('Incremental indexing requires a separate task graph')
    source = base.directory / 'index.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    source_hash = sha256(source)
    if data.get('mode') != 'jina_screenshot' or data['fingerprint'] != fingerprint(base):
        raise ValueError('Source screenshot index does not match its graph')
    if len(data['ids']) != len(set(data['ids'])) or not set(data['ids']) <= set(base.nodes):
        raise ValueError('Source index contains duplicate or unknown pages')
    identity = client.identity()
    if identity != data['encoder']:
        raise ValueError('Jina encoder differs from the source index')
    path = checked_file(base.directory, data['vectors'], data['vectors_sha256'])
    with np.load(path, allow_pickle=False) as archive:
        vectors = {key: archive[key] for key in archive.files}
    for vector in vectors.values():
        normalize(vector)  # Validate shape/values; keep the original arrays byte-for-byte.
    result = copy.deepcopy(data)
    tiles = result.get('tiling')
    expected_keys = set(data['ids'])
    reused = [key for key in data['ids'] if key in graph.nodes]
    for key in data['ids']:
        expected_keys.update(tile['vector'] for tile in (tiles['pages'].get(key, []) if tiles else []))
        if key not in reused:
            continue
        relative = f'snapshots/{key}/page.png'
        checked_file(base.directory, relative, data['images_sha256'][key])
        checked_file(graph.directory, relative, data['images_sha256'][key])
        if page_text(base.nodes[key]) != page_text(graph.nodes[key]):
            raise ValueError('Historical page changed in the task graph')
        for tile in tiles['pages'].get(key, []) if tiles else []:
            crop = checked_file(base.directory, tile['image'], tile['sha256'])
            target = graph.directory / crop.relative_to(base.directory.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(crop, target)
            expected_keys.add(tile['vector'])
    if set(vectors) != expected_keys:
        raise ValueError('Source vectors do not match the indexed pages/views')
    fresh = [key for key in graph.nodes if key not in data['ids']]
    kept_vectors = set(reused)
    for key in reused:
        kept_vectors.update(t['vector'] for t in (tiles['pages'].get(key, []) if tiles else []))
    vectors = {key: vector for key, vector in vectors.items() if key in kept_vectors}
    result['images_sha256'] = {key: result['images_sha256'][key] for key in reused}
    if tiles:
        tiles['pages'] = {key: tiles['pages'].get(key, []) for key in reused}
    log.emit('execution_index_started', reused=len(reused), new_pages=len(fresh), encoder=identity,
             node_scope='all_graph_nodes', excluded_ids=[])
    crop_dir = graph.directory / 'index-crops' / uuid4().hex
    for key in fresh:
        image_path = graph.directory / 'snapshots' / key / 'page.png'
        before = sha256(image_path)
        vectors[key] = client.encode(image_path.resolve(), 'image', identity)
        if tiles:
            tiles['pages'][key] = []
            with Image.open(image_path) as image:
                for number, box in enumerate(tile_boxes(*image.size, tiles['height'], tiles['overlap'])):
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    crop = crop_dir / f'{key}-{number}.png'
                    image.crop(box).save(crop)
                    vector_key = f'{key}__tile_{number}'
                    vectors[vector_key] = client.encode(crop.resolve(), 'image', identity)
                    tiles['pages'][key].append({'kind': 'crop', 'vector': vector_key, 'box': list(box),
                        'image': crop.relative_to(graph.directory).as_posix(), 'sha256': sha256(crop)})
        if sha256(image_path) != before:
            raise ValueError('New screenshot changed during encoding')
        result['images_sha256'][key] = before
        log.emit('jina_index_page', node=key, image_sha256=before, tokens=len(vectors[key]), incremental=True)
    if sha256(source) != source_hash:
        raise ValueError('Source index changed during extension')
    filename = f'jina-vectors-{uuid4().hex}.npz'
    np.savez(graph.directory / filename, **vectors)
    ids = reused + fresh
    result.update(fingerprint=fingerprint(graph), ids=ids, node_scope='all_graph_nodes',
                  texts=[page_text(graph.nodes[k]) for k in ids],
                  endpoint=client.endpoint, vectors=filename, vectors_sha256=sha256(graph.directory / filename),
                  incremental={'source_index_sha256': source_hash, 'reused_ids': reused, 'encoded_ids': fresh,
                               'removed_ids': [key for key in data['ids'] if key not in graph.nodes],
                               'excluded_ids': []})
    # Publish only after every new page/view succeeds. A failed run has no usable partial index.
    write_json(graph.directory / ('index-tiled.json' if tiles else 'index-whole.json'), result)
    write_json(graph.directory / 'index.json', result)
    log.emit('execution_index_completed', reused=len(reused), encoded=len(fresh), pages=len(ids))
    return result
