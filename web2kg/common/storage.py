import hashlib
import json
import time
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from .replay_policy import replay_allowed, executable, mark_input_boundary


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]


def write_json(path, value, *, compact=False, return_hash=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    formatting = {'separators': (',', ':')} if compact else {'indent': 2}
    text = json.dumps(value, ensure_ascii=False, **formatting)
    if return_hash:
        # Hash the exact bytes being published, without a second disk read.
        encoded = text.encode('utf-8')
        temporary.write_bytes(encoded)
    else:
        temporary.write_text(text, encoding="utf-8")
    # Windows readers can briefly prevent replacement of an otherwise writable file.
    for attempt in range(11):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 10:
                raise
            time.sleep(0.1)
    return hashlib.sha256(encoded).hexdigest() if return_hash else None


class EventLog:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "events.jsonl"
        self.lock = threading.Lock()

    def emit(self, event, **data):
        row = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **data}
        with self.lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


class Graph:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.data = {"version": 1, "root": None, "start_url": None, "nodes": {}, "edges": []}
        self._reachable = {}
        self.replay_revision = 0

    def invalidate_reachability(self):
        self._reachable.clear()
        self.replay_revision += 1

    @classmethod
    def load(cls, directory):
        graph = cls(directory)
        graph.data = json.loads((graph.directory / "graph.json").read_text(encoding="utf-8"))
        if graph.data["version"] != 1:
            raise ValueError("Unsupported graph format")
        return graph

    @property
    def nodes(self):
        return self.data["nodes"]

    def save(self):
        return write_json(self.directory / "graph.json", self.data, compact=True, return_hash=True)

    def add_node(self, observation, path):
        from ..stage1.page_admission import admitted
        if not admitted(observation):
            return False
        key = observation["id"]
        if key in self.nodes:
            return False
        node = {k: v for k, v in observation.items() if k not in {"ax", "dom"}}
        node["path"] = path
        node["depth"] = len(path)
        self.nodes[key] = node
        self.invalidate_reachability()
        return True

    def add_edge(self, source, target, action):
        key = digest([source, target, action])
        edge = {"id": key, "source": source, "target": target, "action": action,
                "check": {"state_id": target, "url": self.nodes[target]["url"],
                          "headings": self.nodes[target]["headings"]}}
        mark_input_boundary(edge)
        if not any(e["id"] == key for e in self.data["edges"]):
            self.data["edges"].append(edge)
            self.invalidate_reachability()
        return key

    def edge_replayable(self, edge):
        if edge['action'].get('kind') == 'url_jump':
            entries = self.data.get('url_entries', {})
            return (replay_allowed(edge) and edge['source'] in self.nodes and edge['target'] in self.nodes
                    and edge['source'] in entries and edge['target'] in entries
                    and edge['action'].get('url') == self.nodes[edge['target']]['url']
                    and all(entries[k]['url'] == self.nodes[k]['url'] for k in (edge['source'], edge['target'])))
        return replay_allowed(edge, self.data['start_url'])

    def register_url_entry(self, node, evidence):
        url = self.nodes[node]['url']
        if not executable({'kind': 'url_jump', 'url': url})[0]:
            return False
        entries = self.data.setdefault('url_entries', {})
        entries[node] = {'url': url, 'evidence': evidence}
        root = self.data['root']
        if root in entries and node != root:
            for source, target in ((root, node), (node, root)):
                target_url = self.nodes[target]['url']
                key = self.add_edge(source, target, {'id': digest(['url_jump', target_url]),
                    'kind': 'url_jump', 'role': 'browser', 'name': 'Recorded entry URL', 'url': target_url})
                edge = next(e for e in self.data['edges'] if e['id'] == key)
                edge.update(replayable=True, replay_policy='recorded-entry-v1',
                            replay_reason='URL link between recorded entry initial states', observed_during='entry_registration')
        self.invalidate_reachability()
        return True

    def initialize_url_entries(self):
        if self.data['root'] not in self.nodes:
            return
        self.register_url_entry(self.data['root'], 'graph_root_initial_state')
        # A disconnected node is not by itself proof of a URL-restorable initial state.
        task_start = self.data.get('task_start')
        if task_start in self.nodes:
            self.register_url_entry(task_start, 'recorded_task_initial_state')

    def path(self, source, target):
        if source not in self.nodes or target not in self.nodes:
            return None
        queue = deque([(source, [])])
        visited = {source}
        while queue:
            current, path = queue.popleft()
            if current == target:
                return path
            for edge in self.data["edges"]:
                if edge["source"] == current and edge["target"] not in visited and self.edge_replayable(edge):
                    visited.add(edge["target"])
                    queue.append((edge["target"], path + [edge]))
        return None

    def reachable(self, source=None):
        source = self.data['root'] if source is None else source
        if source in self._reachable:
            return self._reachable[source]
        if source not in self.nodes:
            return set()
        adjacency = {}
        for edge in self.data['edges']:
            if edge['target'] in self.nodes and self.edge_replayable(edge):
                adjacency.setdefault(edge['source'], []).append(edge['target'])
        seen, pending = {source}, deque([source])
        while pending:
            for target in adjacency.get(pending.popleft(), []):
                if target not in seen:
                    seen.add(target)
                    pending.append(target)
        self._reachable[source] = frozenset(seen)
        return self._reachable[source]
