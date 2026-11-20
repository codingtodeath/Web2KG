"""Whole screenshots with optional overlapping crops; never overlays or DOM text."""
import json
from uuid import uuid4
from PIL import Image

import numpy as np

from ..common.jina import JinaClient, maxsim, sha256
from ..common.page_text import fingerprint, page_text
from ..common.storage import write_json


def tile_boxes(width, height, tile_height=900, overlap=128):
    if tile_height < 1 or not 0 <= overlap < tile_height:
        raise ValueError("Require tile_height > overlap >= 0")
    if height <= tile_height:
        return []
    boxes, y = [], 0
    while True:
        boxes.append((0, y, width, min(y + tile_height, height)))
        if y + tile_height >= height:
            return boxes
        y += tile_height - overlap


def score_views(query, vectors, key, tiles=()):
    whole = maxsim(query, vectors[key])
    if not tiles:
        return whole, {"kind": "whole", "score": whole}
    scores = [maxsim(query, vectors[t["vector"]]) for t in tiles]
    best = int(np.argmax(scores))
    # A fixed mix retains page context; max is within each crop, never across crops.
    return (whole + scores[best]) / 2, {**tiles[best], "score": scores[best], "whole_score": whole}


def build_visual_index(graph, client, *, tiles=False, tile_height=900, overlap=128, reuse_whole=False):
    if not graph.nodes:
        raise ValueError("Cannot index an empty graph")
    tile_boxes(1, 1, tile_height, overlap)
    identity = client.identity()
    cached, cached_vectors = None, None
    whole_path = graph.directory / "index-whole.json"
    source = whole_path if whole_path.exists() else graph.directory / "index.json"
    if (tiles or reuse_whole) and source.exists():
        candidate = json.loads(source.read_text(encoding="utf-8"))
        if (candidate.get("mode") == "jina_screenshot" and not candidate.get("tiling")
                and candidate.get("fingerprint") == fingerprint(graph) and candidate.get("encoder") == identity
                and len(candidate['ids']) == len(set(candidate['ids']))
                and set(candidate['ids']) <= set(graph.nodes)):
            vector_path = (graph.directory / candidate["vectors"]).resolve()
            if not vector_path.is_relative_to(graph.directory.resolve()) or sha256(vector_path) != candidate["vectors_sha256"]:
                raise ValueError("Cached whole index changed")
            cached = candidate
            with np.load(vector_path, allow_pickle=False) as archive:
                cached_vectors = {key: archive[key] for key in candidate["ids"]}
    page_tiles = {}
    crop_dir = graph.directory / "index-crops" / uuid4().hex

    ids, vectors, images = list(graph.nodes), {}, {}
    if client.log:
        client.log.emit('index_coverage', node_scope='all_graph_nodes', included=ids)
    for key in ids:
        path = graph.directory / "snapshots" / key / "page.png"
        before = sha256(path)
        if cached and key in cached['ids'] and cached["images_sha256"].get(key) != before:
            raise ValueError("Cached screenshot changed")
        vectors[key] = cached_vectors[key] if cached and key in cached_vectors else client.encode(path.resolve(), "image", identity)
        if tiles:
            with Image.open(path) as image:
                boxes = tile_boxes(*image.size, tile_height, overlap)
                page_tiles[key] = []
                for number, box in enumerate(boxes):
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    crop = crop_dir / f"{key}-{number}.png"
                    image.crop(box).save(crop)
                    vector_key = f"{key}__tile_{number}"
                    vectors[vector_key] = client.encode(crop.resolve(), "image", identity)
                    record = {"kind": "crop", "vector": vector_key, "box": list(box),
                              "image": crop.relative_to(graph.directory).as_posix(), "sha256": sha256(crop)}
                    page_tiles[key].append(record)
                    if client.log:
                        client.log.emit("jina_index_tile", node=key, **record, tokens=len(vectors[vector_key]))
        if sha256(path) != before:
            raise ValueError("Screenshot changed during encoding")
        images[key] = before
        if client.log:
            client.log.emit("jina_index_page", node=key, image_sha256=before, tokens=len(vectors[key]))
    filename = f"jina-vectors-{uuid4().hex}.npz"
    np.savez(graph.directory / filename, **vectors)
    data = {"version": 2, "mode": "jina_screenshot", "fingerprint": fingerprint(graph),
            "ids": ids, "node_scope": "all_graph_nodes", "texts": [page_text(graph.nodes[key]) for key in ids],
            "text_role": "display_only_not_embedded", "provider": "local-jina",
            "model": "jina-embeddings-v4", "encoder": identity, "endpoint": client.endpoint,
            "dimension": 128, "score": "mean_token_max_cosine", "vectors": filename,
            "vectors_sha256": sha256(graph.directory / filename), "images_sha256": images}
    if tiles:
        whole_filename = f"jina-whole-{uuid4().hex}.npz"
        np.savez(graph.directory / whole_filename, **{key: vectors[key] for key in ids})
        cached = {**data, "vectors": whole_filename,
                  "vectors_sha256": sha256(graph.directory / whole_filename)}
    if tiles:
        data["score"] = "half_whole_half_best_crop"
        data["tiling"] = {"height": tile_height, "overlap": overlap,
                          "score": "0.5*whole+0.5*max_crop_mean_maxsim", "pages": page_tiles}
        if cached:
            write_json(whole_path, cached)
        write_json(graph.directory / "index-tiled.json", data)
    else:
        write_json(whole_path, data)
    write_json(graph.directory / "index.json", data)
    return data


def retrieve_visual(graph, query, top_k, client=None, *, allowed_ids=None):
    data = json.loads((graph.directory / "index.json").read_text(encoding="utf-8"))
    if data['fingerprint'] != fingerprint(graph):
        raise ValueError('Graph changed; rebuild its index')
    from ..stage2.retrieval import candidate_positions, template_order
    positions = candidate_positions(graph, data, allowed_ids)
    if not positions:
        return []
    path = (graph.directory / data["vectors"]).resolve()
    if not path.is_relative_to(graph.directory.resolve()) or sha256(path) != data["vectors_sha256"]:
        raise ValueError("Screenshot index vectors changed; rebuild the index")
    for key, expected in data["images_sha256"].items():
        if sha256(graph.directory / "snapshots" / key / "page.png") != expected:
            raise ValueError("Indexed screenshot changed; rebuild the index")
    client = client or JinaClient(data["endpoint"])
    if client.identity() != data["encoder"]:
        raise ValueError("Jina encoder differs from the saved index")
    query_vector = client.encode(query, "text", data["encoder"])
    with np.load(path, allow_pickle=False) as vectors:
        scored = [score_views(query_vector, vectors, key, data.get("tiling", {}).get("pages", {}).get(key, []))
                  for key in [data['ids'][i] for i in positions]]
        scores = [row[0] for row in scored]
    order = template_order(graph, [data['ids'][i] for i in positions], scores, top_k)
    return [{"id": data["ids"][positions[i]], "score": scores[i], "text": data["texts"][positions[i]],
             "title": graph.nodes[data["ids"][positions[i]]]["title"], "url": graph.nodes[data["ids"][positions[i]]]["url"],
             "index_mode": data["mode"], "text_role": data["text_role"], "matched_view": scored[i][1]} for i in order]


def activate_visual_index(graph, client, variant):
    if variant not in {"whole", "tiled"}:
        raise ValueError("Unknown screenshot index variant")
    data = json.loads((graph.directory / f"index-{variant}.json").read_text(encoding="utf-8"))
    if data["fingerprint"] != fingerprint(graph) or data["encoder"] != client.identity():
        raise ValueError("Saved index graph or encoder differs")
    path = (graph.directory / data["vectors"]).resolve()
    if not path.is_relative_to(graph.directory.resolve()) or sha256(path) != data["vectors_sha256"]:
        raise ValueError("Saved index vectors changed")
    for key, expected in data["images_sha256"].items():
        if sha256(graph.directory / "snapshots" / key / "page.png") != expected:
            raise ValueError("Indexed screenshot changed")
    write_json(graph.directory / "index.json", data)
    return data
