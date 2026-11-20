"""Programmatic live matching and a bounded, advisory template neighborhood."""
import json
import time
from pathlib import Path

from ..common.replay_policy import replay_allowed
from ..stage1.function_cards import file_hash, load_layer
from ..stage1.templates import control_profile, local_variant, match_candidate, page_similarity, structure_profile
from ..stage1.templates import optional_entry_candidates, uses_optional_entry
from ..stage1.control_structure import ControlContext, control_comparison
from ..stage1.page_structure import EmptyStructureError
from .retrieval import tokens


class TemplateKnowledge:
    def __init__(self, graph, cards_path, log):
        self.graph, self.log = graph, log
        data = json.loads(Path(cards_path).read_text(encoding="utf-8"))
        self.cards_path = Path(cards_path).resolve()
        source = data["source"]
        path, self.layer = load_layer(graph, graph.directory / source["layer"])
        if source["graph_sha256"] != file_hash(graph.directory / "graph.json") or source["layer_sha256"] != file_hash(path):
            raise ValueError("Function cards are stale for this graph/template layer")
        self.cards = {k: row["card"] for k, row in data["cards"].items() if row["status"] == "completed"}
        self.groups = {g["id"]: g for g in self.layer["templates"]}
        self.previous = self.previous_match = self.pending_action = None
        self.profile_cache = {}

    def candidate(self, key):
        template = self.layer["assignments"][key]["template"]
        return {"template_id": template, "function_card": self.cards.get(template),
                "scope": "Historical sampled template; functions may differ on this member."}

    def after_action(self, action):
        self.pending_action = action

    def replay_action(self, source, action):
        self.previous_match = self.match(source)
        self.previous, self.pending_action = source, action

    def profile(self, observation):
        key = observation["id"]
        if key not in self.profile_cache:
            try:
                self.profile_cache[key] = structure_profile(observation["ax"], observation["dom"])
            except EmptyStructureError:
                self.profile_cache[key] = None
        return self.profile_cache[key]

    def match(self, observation):
        started = time.perf_counter()
        saved = self.layer['assignments'].get(observation['id'])
        if saved and self.groups[saved['template']].get('isolated'):
            result = {'template_id': saved['template'], 'reason': 'isolated_saved_state', 'score': None}
            self.log.emit('template_match', node=observation['id'], **result)
            return result
        if self.profile(observation) is None:
            result = {"template_id": None, "reason": "no_visible_structure", "score": None,
                      "milliseconds": round((time.perf_counter() - started) * 1000, 3)}
            self.log.emit("template_match", node=observation["id"], **result)
            return result
        assignments, profiles = self.layer["assignments"], self.layer["profiles"]
        key = observation["id"]
        if key in assignments and observation["url"] == self.graph.nodes[key]["url"]:
            result = {"template_id": assignments[key]["template"], "reason": "saved_state_identity", "score": None}
        elif self.previous and key == self.previous["id"]:
            result = {**self.previous_match, "reason": "unchanged_live_state"}
        else:
            current = self.profile(observation)
            parent = self.previous_match.get("template_id") if self.previous_match else None
            action = self.pending_action
            entry = control_profile(action, ControlContext(self.previous["ax"], self.previous["dom"])) if (
                self.previous and action) else {"available": False, "kind": None, "role": None}
            modern = uses_optional_entry(self.layer)
            candidates = [{**{k: v for k, v in c.items() if k != 'template'}, 'template_id': c['template']}
                          for c in optional_entry_candidates(self.layer, current, parent, entry)] if modern else []
            page_features = self.layer['parameters']['page_features']
            for template, group in ([] if modern else self.groups.items()):
                if group.get('isolated'):
                    continue
                reference = assignments[group["representative"]]
                # A live page reached through an action is not a new root.
                if reference["parent_node"] is None:
                    continue
                other_parent = reference["parent_template"]
                parent_score = page_similarity(profiles[self.groups[parent]["representative"]],
                    profiles[self.groups[other_parent]["representative"]], page_features) if parent and other_parent else 0.
                scores = {"parent": parent_score, "incoming": control_comparison(entry, reference["incoming"])["score"],
                          "page": page_similarity(current, profiles[group["representative"]], page_features)}
                candidates.append({"template_id": template, "scores": scores,
                                   **match_candidate(scores, self.layer['parameters'])})
            if not modern:
                candidates.sort(key=lambda c: (-c["score"], c["template_id"]))
            best = next((c for c in candidates if c['eligible']), None)
            inherited = parent and action and not self.groups.get(parent, {}).get('isolated') and local_variant(self.previous, observation, action,
                                                            self.profile(self.previous), current)
            if inherited:
                result = {"template_id": parent, "reason": "observed_local_state_variant", "score": None}
            elif best:
                result = {**best, "reason": "weighted_match"}
            else:
                result = {"template_id": None, "reason": "unknown_template",
                          "score": candidates[0]['score'] if candidates else 0.}
            result["closest"] = candidates[:3]
        result["milliseconds"] = round((time.perf_counter() - started) * 1000, 3)
        self.log.emit("template_match", node=key, **result)
        return result

    def observe(self, observation, task):
        match = self.match(observation)
        template = match["template_id"]
        hints = []
        if template:
            for edge in self.graph.data["edges"]:
                if (edge['action'].get('kind') == 'url_jump' or not replay_allowed(edge, self.graph.data['start_url']) or edge['source'] not in self.layer['assignments']
                        or edge['target'] not in self.layer['assignments']):
                    continue
                if self.layer["assignments"][edge["source"]]["template"] != template:
                    continue
                saved = edge["action"]
                live = [a for a in observation["actions"] if a["role"] == saved["role"] and a["name"] == saved["name"]
                        and replay_allowed({**edge, 'action': a}, self.graph.data['start_url'])]
                if not live:
                    continue
                target = self.layer["assignments"][edge["target"]]["template"]
                card = self.cards.get(target, {})
                row = {"target_template": target, "target_name": card.get("name"),
                       "target_summary": card.get("summary", ""), "control_label": saved["name"],
                       "current_control_ids": [a.get("ax_node_id", a["id"]) for a in live], "evidence_edge": edge["id"],
                       "scope": "Same label/role only; historical destination is not guaranteed."}
                if not any(h["target_template"] == target and h["control_label"] == saved["name"] for h in hints):
                    hints.append(row)
            query = set(tokens(task))
            hints.sort(key=lambda h: -len(query & set(tokens(h["target_summary"] + " " + h["control_label"]))))
        context = {"match": match, "function_card": self.cards.get(template), "nearby": hints[:6],
                   "nearby_total": len(hints), "notice": "Hints, not permissions or proof of identity/completion. "
                   "Use only actual current controls. Unknown template: continue from the live page; do not invent a card."}
        self.log.emit("template_context", node=observation["id"], context=context)
        self.previous, self.previous_match, self.pending_action = observation, match, None
        return context
