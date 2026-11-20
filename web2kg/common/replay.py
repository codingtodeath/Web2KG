import time

from playwright.sync_api import Error as BrowserError

from .replay_policy import INPUT_KINDS
from .diagnostics import page_ref, record_failure
from .interaction import VERSION, state_id, state_matches


class RootStateMismatch(ValueError):
    """Reset no longer reaches the recorded entry; further root restores cannot proceed."""


class SourceRestoreError(ValueError):
    """The recorded path cannot restore this source; its other candidates cannot run."""


def verify(browser, expected, saved=None):
    resolver = getattr(browser, 'state_resolver', None)
    if saved is None and resolver is not None:
        saved = resolver.graph.nodes.get(expected)
    observed = browser.observe()
    for attempt in range(4):
        matches = (state_id(saved) == expected and state_matches(saved, observed)
                   if saved and saved.get('identity_version') == VERSION
                   else observed['id'] == expected)
        if not matches and resolver is not None:
            matches = resolver.matches(expected, observed)
        if matches:
            return observed
        if attempt < 3:
            browser.page.wait_for_timeout(200)
            observed = browser.observe()
    if resolver is not None:
        observed = resolver.resolve(observed, learn=True, expected=expected)
        if resolver.matches(expected, observed):
            return observed
    raise ValueError(f"State mismatch: expected {expected}, observed {observed['id']}")


def bind_resolver(browser, graph):
    """Loaded graphs reuse learned rules during navigation without requiring a judge."""
    resolver = getattr(browser, 'state_resolver', None)
    if resolver is not None and resolver.graph is graph:
        return
    if graph.data.get('state_equivalence'):
        from .state_equivalence import StateResolver
        browser.state_resolver = StateResolver(graph, log=browser.log)
    elif resolver is not None:
        browser.state_resolver = None


def execute_path(browser, graph, path, log, strategy="role", on_action=None, poll=None):
    bind_resolver(browser, graph)
    started = time.monotonic()
    for index, edge in enumerate(path):
        if poll:
            poll()
        phase, action_attempted = "source_check", False
        try:
            source = verify(browser, edge["source"], graph.nodes.get(edge['source']))
            phase = "policy_check"
            if poll:
                poll()
            if edge['action'].get('kind') in INPUT_KINDS:
                log.emit('replay_input_required', edge=edge['id'], source=edge['source'], steps=index)
                return {'success': False, 'status': 'requires_input', 'reason': 'actor_input_required',
                        'error': 'Actor input required before this transition', 'steps': index,
                        'seconds': time.monotonic() - started}
            if not graph.edge_replayable(edge):
                raise ValueError('Execution-only transition cannot be replayed')
            phase, action_attempted = "action", True
            if edge['action'].get('kind') == 'url_jump':
                browser.page.goto(edge['action']['url'], wait_until='domcontentloaded', timeout=20000)
                browser.settle()
            else:
                browser.act(edge["action"], strategy)
            if on_action:
                on_action(source, edge["action"])
            phase = "target_check"
            observed = verify(browser, edge["check"]["state_id"], graph.nodes.get(edge['target']))
            if observed["url"] != edge["check"]["url"] or observed["headings"] != edge["check"]["headings"]:
                raise ValueError("Transition postcondition mismatch")
            log.emit("replay_step", edge=edge["id"], step=index, strategy=strategy,
                     source=edge["source"], target=edge["target"], success=True,
                     provisional=edge.get('replay_policy') == 'construction-pending-v1',
                     source_page=page_ref(graph.nodes[edge["source"]]), action=edge["action"])
        except (BrowserError, ValueError) as error:
            record_failure(browser, log, "replay_step", graph.nodes[edge["source"]], edge["action"],
                           error, phase, action_attempted, edge=edge["id"], step=index, strategy=strategy,
                           source=edge["source"], target=edge["target"], success=False)
            return {"success": False, "failed_step": index, "error": str(error),
                    "steps": index, "seconds": time.monotonic() - started}
    return {"success": True, "failed_step": None, "steps": len(path), "seconds": time.monotonic() - started}


def restore(browser, graph, target, log, strategy="role", poll=None):
    bind_resolver(browser, graph)
    if poll:
        poll()
    browser.reset()
    try:
        verify(browser, graph.data["root"], graph.nodes.get(graph.data['root']))
    except ValueError as error:
        raise RootStateMismatch(str(error)) from error
    path = graph.path(graph.data["root"], target)
    if path is None:
        raise SourceRestoreError("Source is unreachable from the root")
    result = execute_path(browser, graph, path, log, strategy, poll=poll)
    if not result["success"]:
        raise SourceRestoreError("Source restore failed: " + result["error"])
    try:
        verify(browser, target, graph.nodes.get(target))
    except ValueError as error:
        raise SourceRestoreError(str(error)) from error
    log.emit("source_restored", node=target, steps=len(path))
