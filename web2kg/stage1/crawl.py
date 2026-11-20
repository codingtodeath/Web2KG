from collections import deque
from dataclasses import asdict, dataclass
import json
import sys
import shutil
import hashlib
from pathlib import Path

from playwright.sync_api import Error as BrowserError

from ..common.replay import RootStateMismatch, SourceRestoreError, bind_resolver, restore, verify
from ..common.diagnostics import record_failure
from .selection import select_actions
from .url_pruning import known_targets, record_pruned
from .page_admission import admitted
from ..common.storage import Graph


@dataclass
class CrawlBudget:
    max_states: int = 25
    max_depth: int = 2
    actions_per_page: int = 12
    per_group: int = 1
    max_attempts: int = 100

    def __post_init__(self):
        if min(self.max_states, self.actions_per_page, self.per_group, self.max_attempts) < 1 or self.max_depth < 0:
            raise ValueError("Budgets must be positive; depth may be zero")


def crawl(browser, directory, log, budget=None, model=None, resume=False, explorer=None, state_judge=None, seed=None,
          recover_annotations=False):
    budget = budget or CrawlBudget()
    pipeline = explorer if hasattr(explorer, 'prefetch') else None
    if pipeline and resume:
        raise ValueError('Parallel construction requires a new output directory')
    graph = Graph(directory)
    if seed is not None:
        seed = Path(seed).resolve()
        destination = Path(directory).resolve()
        if resume or destination == seed or destination.is_relative_to(seed) or seed.is_relative_to(destination):
            raise ValueError('Expansion requires an independent new directory, not resume')
        if (destination / 'graph.json').exists():
            raise ValueError('Expansion output already contains a graph')
        original = Graph.load(seed)
        if (original.data['start_url'] != browser.start_url or
                original.data.get('identity_mode') != browser.identity_mode or
                original.data.get('viewport') != browser.viewport):
            raise ValueError('Expansion requires the seed browser identity settings')
        if any(r.get('annotation', {}).get('status') == 'pending' for r in original.data.get('transition_records', [])):
            raise ValueError('Cannot expand a graph with unresolved annotations')
        for item in seed.iterdir():
            if item.name == 'events.jsonl':
                continue
            elif item.is_dir():
                shutil.copytree(item, destination / item.name)
            else:
                shutil.copy2(item, destination / item.name)
        history = destination / 'seed-event-history'
        history.mkdir(exist_ok=True)
        if (seed / 'events.jsonl').exists():
            key = hashlib.sha256((seed / 'graph.json').read_bytes()).hexdigest()
            shutil.copy2(seed / 'events.jsonl', history / (key + '.jsonl'))
        graph = Graph.load(destination)
        from ..common.storage import write_json
        write_json(destination / 'seed-rankings.json', graph.data.get('rankings', {}))
        graph.data['rankings'] = {}
        graph.data['expansion'] = dict(seed=str(seed), inherited_nodes=len(graph.nodes),
            inherited_edges=len(graph.data['edges']), inherited_attempts=graph.data.get('attempts', 0),
            inherited_records=len(graph.data.get('transition_records', [])))
        graph.data.update(budget=asdict(budget), llm_explorer=explorer.identity if explorer else None)
    bind_resolver(browser, graph)
    if state_judge is not None:
        if resume or browser.identity_mode != 'interaction':
            raise ValueError('Learning equivalence requires a fresh interaction crawl')
        from ..common.state_equivalence import StateResolver
        browser.state_resolver = StateResolver(graph, state_judge,
            reserve=pipeline.reserve_external if pipeline else None, log=log)
    attempted, parents = set(), {}
    if seed is not None:
        attempts = graph.data.get('attempts', 0)
        attempted = {(r['source'], r['action']['id']) for r in graph.data.get('transition_records', [])}
        # Restore the recorded root; never silently replace it with today's homepage.
        restore(browser, graph, graph.data['root'], log)
        queue = deque(sorted(graph.nodes, key=lambda key: (graph.nodes[key]['depth'], key)))
        graph.save()
        log.emit('crawl_expansion_started', **graph.data['expansion'], budget=asdict(budget))
    elif resume:
        graph = Graph.load(directory)
        bind_resolver(browser, graph)
        if (model is not None or graph.data.get("review_mode") != ("llm_ranked" if explorer else "rules")
                or graph.data.get("llm_explorer") != (explorer.identity if explorer else None)
                or graph.data["budget"] != asdict(budget)
                or graph.data["start_url"] != browser.start_url
                or graph.data.get("identity_mode") != browser.identity_mode
                or graph.data.get("viewport") != browser.viewport):
            raise ValueError("Resume requires the original crawl settings")
        attempts = graph.data.get("attempts", 0)
        for line in (graph.directory / "events.jsonl").read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event["event"] == "crawl_finished":
                raise ValueError("Completed crawls cannot be resumed")
            if event["event"] in {"crawl_attempt_started", "exploration_action_started", "exploration_failed"}:
                attempted.add((event["source"], event["action"]["id"]))
                attempts = max(attempts, event.get("attempt", 0))
        for key, node in graph.nodes.items():
            if node["path"]:
                sources = {edge["source"] for edge in graph.data["edges"]
                           if edge["target"] == key and edge["action"] == node["path"][-1]
                           and graph.nodes[edge["source"]]["path"] == node["path"][:-1]}
                if len(sources) != 1:
                    raise ValueError("Cannot reconstruct the original discovery parent")
                parents[key] = sources.pop()
        restore(browser, graph, graph.data["root"], log)
        verify(browser, graph.data["root"])
        queue = deque(sorted(graph.nodes, key=lambda key: graph.nodes[key]["depth"]))
        log.emit("crawl_resumed", states=len(graph.nodes), attempts=attempts,
                 skipped_attempted_actions=len(attempted), budget=asdict(budget))
    elif (graph.directory / "graph.json").exists():
        raise ValueError("Output already contains a graph; choose a new run directory")
    else:
        root = browser.observe(directory)
        graph.data.update(root=root["id"], start_url=browser.start_url, budget=asdict(budget),
                      review_mode="llm_ranked" if explorer else "llm" if model else "rules",
                      llm_explorer=explorer.identity if explorer else None, identity_mode=browser.identity_mode,
                      viewport=browser.viewport, exploration="llm_ranked_bfs" if explorer else "adaptive_bfs" if browser.identity_mode == "webnavigator" else "grouped_bfs")
        graph.add_node(root, [])
        if root['id'] not in graph.nodes:
            graph.data['root'] = None
            graph.save()
            log.emit('crawl_finished', states=0, edges=0, attempts=0, reason='blank_or_loading_initial_page')
            if pipeline:
                pipeline.close(graph)
            return graph
        graph.save()
        log.emit("crawl_started", root=root["id"], budget=asdict(budget))
        queue, attempts = deque([root["id"]]), 0
    attempt_limit = attempts + budget.max_attempts if seed is not None else budget.max_attempts
    failed = False
    try:
        if recover_annotations:
            if seed is None or pipeline is None:
                raise ValueError('Annotation recovery requires a new seeded parallel crawl')
            pipeline.recover_annotations(graph)
        while queue and len(graph.nodes) < budget.max_states and attempts < attempt_limit:
            if pipeline:
                pipeline.poll(graph)
            source = queue.popleft()
            if graph.path(graph.data['root'], source) is None:
                log.emit('crawl_unreachable_skipped', node=source, reason='no_replayable_path')
                continue
            node = graph.nodes[source]
            if node["depth"] >= budget.max_depth:
                continue
            eligible = None
            if browser.identity_mode == "webnavigator" and explorer is None:
                parent_paths = graph.nodes[parents[source]]["structural_paths"] if source in parents else []
                eligible = set(node["structural_paths"]) - set(parent_paths)
                log.emit("structural_difference", node=source, parent=parents.get(source),
                         total_paths=len(node["structural_paths"]), new_paths=len(eligible))
            selected = explorer.select(graph, node, budget.actions_per_page) if explorer else select_actions(node, browser.start_url, log, budget.actions_per_page,
                                      budget.actions_per_page if eligible is not None else budget.per_group,
                                      model, eligible_paths=eligible)
            states = {action["id"]: "attempted" if (source, action["id"]) in attempted else "selected"
                      for action in selected}
            pruned = {(r["source"], r["action"]["id"]) for r in graph.data.get("pruned_links", [])}
            def pending_actions(rest):
                return [dict(source=source, **{k: a[k] for k in ('id', 'kind', 'role', 'name', 'href') if k in a})
                        for a in rest if (source, a['id']) not in attempted]
            if pipeline:
                pipeline.prefetch(graph, queue, pending_actions(selected))
            source_failed = False
            for action_index, action in enumerate(selected):
                if pipeline:
                    pipeline.before_attempt(graph)
                if (source, action["id"]) in attempted or (source, action["id"]) in pruned:
                    continue
                targets = known_targets(graph, node, action) if explorer else []
                if targets:
                    record_pruned(graph, node, action, targets, log, "before_attempt")
                    graph.save()
                    continue
                if len(graph.nodes) >= budget.max_states or attempts >= attempt_limit:
                    break
                attempts += 1
                attempted.add((source, action["id"]))
                log.emit("crawl_attempt_started", source=source, action=action, attempt=attempts)
                phase, action_attempted = "restore_source", False
                target, edge_id, outcome, failure = None, None, "not_attempted", None
                before = None
                is_new, before_image, marker = False, None, None
                was_reachable = False
                try:
                    restore(browser, graph, source, log, **({'poll': lambda: pipeline.poll(graph)} if pipeline else {}))
                    before = verify(browser, source)
                    if pipeline:
                        pipeline.check_stop()
                    states[action["id"]] = "attempted"
                    try:
                        if pipeline:
                            evidence = Path(directory) / 'transitions' / f'{attempts:05d}-evidence'
                            metadata = browser.save_annotation({**before, 'actions': [{**action, 'number': 1}]},
                                                               evidence, {action['id']: 'selected'})
                            before_image = evidence / 'page-annotated.png'
                            marker = {'number': 1, 'status': metadata['items'][0]['status']}
                        else:
                            browser.save_annotation(node, Path(directory) / "snapshots" / source, states)
                    except (BrowserError, OSError, ValueError) as error:
                        log.emit("annotation_failed", node=source, error=str(error))
                        if pipeline:
                            raise RuntimeError('Cannot capture actual interaction screenshot') from error
                    phase, action_attempted = "action", True
                    log.emit("exploration_action_started", source=source, action=action, attempt=attempts)
                    browser.act(action)
                    phase = "capture_result"
                    target = browser.observe()
                    resolver = getattr(browser, 'state_resolver', None)
                    if resolver is not None:
                        target = resolver.resolve(target, learn=True)
                    if not admitted(target):
                        outcome = 'unindexed_transition'
                        log.emit('exploration_state_excluded', node=target['id'], reason=target['page_content']['kind'])
                        continue
                    if target["id"] == source:
                        outcome = "no_change"
                        log.emit("exploration_no_change", source=source, action=action)
                        continue
                    is_new = target["id"] not in graph.nodes
                    was_reachable = target['id'] in graph.reachable()
                    if is_new:
                        browser.save_observation(target, directory)
                        graph.add_node(target, node["path"] + [action])
                        parents[target["id"]] = source
                        queue.append(target["id"])
                    edge_id = graph.add_edge(source, target["id"], action)
                    outcome = "transition"
                    graph.save()
                    log.emit("transition_discovered", source=source, target=target["id"],
                             edge=edge_id, action=action, new_state=is_new, states=len(graph.nodes))
                except (BrowserError, ValueError) as error:
                    outcome = "failed" if action_attempted else "not_attempted"
                    failure = str(error).splitlines()[0]
                    record_failure(browser, log, "exploration_failed", node, action, error, phase,
                                   action_attempted, source=source, attempt=attempts)
                    if explorer:
                        try:
                            target = browser.observe()
                        except (BrowserError, ValueError):
                            target = None
                    if isinstance(error, RootStateMismatch):
                        log.emit('crawl_root_changed', expected_root=graph.data['root'],
                                 observed=target['id'] if target else None, attempt=attempts)
                        raise  # Keep this failure record and drain annotations; do not try every candidate.
                    if isinstance(error, SourceRestoreError):
                        source_failed = True
                        skipped = pending_actions(selected[action_index + 1:])
                        receipt = dict(source=source, failed_attempt=attempts,
                                       skipped_candidate_ids=[a['id'] for a in skipped],
                                       reason='source_restore_failed')
                        graph.data.setdefault('unavailable_sources', []).append(receipt)
                        log.emit('crawl_source_unavailable', **receipt)
                finally:
                    if pipeline:
                        images = ([before_image, Path(directory) / 'snapshots' / target['id'] / 'page.png']
                                  if before_image and target else [])
                        pipeline.record(graph, node, action, target, attempts, outcome, phase,
                            action_attempted, edge_id, failure, new_state=is_new, images=images, mark=marker)
                        if edge_id and not is_new and not was_reachable and target['id'] in graph.reachable():
                            queue.append(target['id'])
                        if not source_failed and sys.exc_info()[0] is None and len(graph.nodes) < budget.max_states and attempts < attempt_limit:
                            pipeline.prefetch(graph, queue, pending_actions(selected[action_index+1:]))
                    elif explorer:
                        explorer.describe(graph, node, action, target, attempts, outcome, phase,
                                          action_attempted, edge_id, failure, before=before)
                        if edge_id and not is_new and not was_reachable and target['id'] in graph.reachable():
                            queue.append(target['id'])
                if source_failed:
                    graph.save()
                    break
        if pipeline:
            pipeline.close(graph)
    except (RuntimeError, ValueError, OSError) as error:
        failed = True
        log.emit("crawl_interrupted", error=str(error), error_type=type(error).__name__,
                 states=len(graph.nodes), attempts=attempts, retry=False)
        raise
    finally:
        try:
            if pipeline:
                pipeline.close(graph, failed=failed)
        finally:
            graph.data["attempts"] = attempts
            graph.save()
    reason = "state_budget" if len(graph.nodes) >= budget.max_states else (
        "attempt_budget" if attempts >= attempt_limit else "frontier_exhausted")
    log.emit("crawl_finished", states=len(graph.nodes), edges=len(graph.data["edges"]), attempts=attempts, reason=reason)
    return graph
