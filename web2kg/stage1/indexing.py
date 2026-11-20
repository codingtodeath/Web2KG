import numpy as np

from ..common.page_text import fingerprint, page_text
from ..common.storage import digest, write_json


def build_index(graph, model=None):
    ids = list(graph.nodes)
    if not ids:
        raise ValueError('No graph pages to index')
    texts = [page_text(graph.nodes[key]) for key in ids]
    data = {"version": 1, "fingerprint": fingerprint(graph), "ids": ids, "texts": texts,
            "mode": "embedding" if model else "lexical", "node_scope": "all_graph_nodes"}
    if model:
        vectors = model.embed(texts)
        data.update(provider=model.config.provider, model=model.config.model, dimension=vectors.shape[1])
        # Generation-specific filenames keep the previous manifest valid if indexing is interrupted.
        data["vectors"] = f"vectors-{digest([data['fingerprint'], data['model'], data['provider']])}.npy"
        np.save(graph.directory / data["vectors"], vectors, allow_pickle=False)
    write_json(graph.directory / "index.json", data)
    return data
