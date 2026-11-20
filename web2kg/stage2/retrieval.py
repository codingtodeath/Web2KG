import json
import math
import re
from collections import Counter

import numpy as np

from ..common.page_text import fingerprint


def candidate_positions(graph, data, allowed_ids=None):
    # A task's source index is immutable. Recompute only after graph/index changes.
    # Offline queries default to root; live callers supply current-state reachability.
    allowed = None if allowed_ids is None else frozenset(allowed_ids)
    key = (graph.replay_revision, tuple(data['ids']), allowed)
    cached = getattr(graph, '_candidate_positions', None)
    if cached is None or cached[0] != key:
        reachable = graph.reachable() if allowed is None else allowed
        positions = [i for i, node in enumerate(data['ids']) if node in graph.nodes and node in reachable]
        graph._candidate_positions = (key, positions)
    return graph._candidate_positions[1]


def tokens(text):
    english = re.findall(r"[a-z0-9]+", text.lower())
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    return english + chinese + ["".join(chinese[i:i + 2]) for i in range(len(chinese) - 1)]


def template_order(graph, ids, scores, top_k):
    """Rank the full candidate pool before retaining one representative per template."""
    pointer = graph.directory / 'templates/active.json'
    stamp = pointer.read_bytes() if pointer.exists() else None
    cache = getattr(graph, '_retrieval_templates', None)
    if cache is None or cache[0] != (graph.replay_revision, stamp):
        assignments = {}
        if stamp:
            location = json.loads(stamp)['directory']
            layer = json.loads((graph.directory / location / 'templates.json').read_text(encoding='utf-8'))
            assignments = {k: v['template'] for k, v in layer['assignments'].items()}
        graph._retrieval_templates = ((graph.replay_revision, stamp), assignments)
    assignments = graph._retrieval_templates[1]
    seen, result = set(), []
    for i in np.argsort(-np.asarray(scores), kind='stable'):
        template = assignments.get(ids[i], graph.nodes[ids[i]].get('template_id'))
        key = ('template', template) if template is not None else ('page', ids[i])
        if key not in seen:
            seen.add(key)
            result.append(int(i))
            if len(result) == top_k:
                break
    return result


def lexical_vectors(documents, query):
    counts = [Counter(tokens(text)) for text in documents]
    vocabulary = sorted(set().union(*(row.keys() for row in counts)))
    idf = {word: math.log((1 + len(counts)) / (1 + sum(word in row for row in counts))) + 1
           for word in vocabulary}
    rows = counts + [Counter(tokens(query))]
    array = np.asarray([[math.log1p(row[word]) * idf[word] for word in vocabulary] for row in rows])
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    array = array / np.maximum(norms, 1e-12)
    return array[:-1], array[-1]


def retrieve(graph, query, model=None, top_k=5, *, allowed_ids=None):
    if top_k < 1 or not query.strip():
        raise ValueError("Query must be nonempty and top_k positive")
    data = json.loads((graph.directory / "index.json").read_text(encoding="utf-8"))
    if data["fingerprint"] != fingerprint(graph):
        raise ValueError("Graph changed; rebuild its index")
    positions = candidate_positions(graph, data, allowed_ids)
    if not positions:
        return []
    if data["mode"] == "jina_screenshot":
        from ..stage1.visual_index import retrieve_visual
        return retrieve_visual(graph, query, top_k, model, allowed_ids=allowed_ids)
    if data["mode"] == "embedding":
        if not model or (model.config.provider, model.config.model) != (data["provider"], data["model"]):
            raise ValueError("Query embedding model must match the saved index")
        vectors = np.load(graph.directory / data["vectors"], allow_pickle=False)
        query_vector = model.embed([query])[0]
        if vectors.shape != (len(data["ids"]), len(query_vector)):
            raise ValueError("Index vector dimensions do not match")
    else:
        vectors, query_vector = lexical_vectors(data["texts"], query)
    scores = vectors[positions] @ query_vector
    order = template_order(graph, [data['ids'][i] for i in positions], scores, top_k)
    scores = dict(zip(positions, scores))
    order = [positions[i] for i in order]
    return [{"id": data["ids"][i], "score": float(scores[i]), "text": data["texts"][i],
             "title": graph.nodes[data["ids"][i]]["title"], "url": graph.nodes[data["ids"][i]]["url"]}
            for i in order]


def choose(query, candidates, model, log):
    result = model.json(
        'Choose a functional entry page for the user task. A page may be a useful starting point without completing the task. '
        'Return {"node_id": "an exact candidate id or null", "reason": "brief explanation"}. '
        'Choose null when none is useful; never invent an id. Similarity scores are not confidence probabilities.',
        {"task": query, "candidates": candidates},
    )
    if "node_id" not in result:
        raise ValueError("Selector omitted node_id")
    selected = result["node_id"]
    if selected is not None and selected not in {c["id"] for c in candidates}:
        raise ValueError("Selector returned an unknown candidate")
    log.emit("candidate_chosen", task=query, **result)
    return selected
