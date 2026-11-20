import json
from collections import Counter
from pathlib import Path

import pytest

from web2kg.common.browser import Browser
from web2kg.common.storage import EventLog, Graph, write_json
from web2kg.stage1.crawl import CrawlBudget, crawl


def test_atomic_replace_handles_temporary_lock_and_preserves_old_file(tmp_path, monkeypatch):
    path = tmp_path / 'data.json'
    write_json(path, {'value': 'old'})
    replace = Path.replace
    locks = [True, True, False]

    def transient(self, target):
        if locks.pop(0):
            raise PermissionError('reader holds file')
        return replace(self, target)

    monkeypatch.setattr('web2kg.common.storage.time.sleep', lambda _: None)
    monkeypatch.setattr(Path, 'replace', transient)
    write_json(path, {'value': 'new'})
    assert json.loads(path.read_text()) == {'value': 'new'}

    def permanent(self, target):
        raise PermissionError('persistent access denied')

    monkeypatch.setattr(Path, 'replace', permanent)
    with pytest.raises(PermissionError):
        write_json(path, {'value': 'must not publish'})
    assert json.loads(path.read_text()) == {'value': 'new'}


def test_interrupted_crawl_resumes_without_repeating_attempts(site, tmp_path, monkeypatch):
    directory = tmp_path / 'graph'
    log, budget = EventLog(directory), CrawlBudget(10, 2, 12, 1, 18)
    original_save = Graph.save
    failed = False

    def interrupted_save(graph):
        nonlocal failed
        if len(graph.nodes) >= 3 and not failed:
            failed = True
            raise OSError('simulated interrupted save')
        original_save(graph)

    monkeypatch.setattr(Graph, 'save', interrupted_save)
    with Browser(site, log, identity_mode='webnavigator', settle_ms=20) as browser:
        with pytest.raises(OSError):
            crawl(browser, directory, log, budget)
    assert failed
    saved_images = {p: p.read_bytes() for p in directory.glob('snapshots/*/page.png')}
    with pytest.raises(ValueError, match='original'):
        crawl(browser, directory, log, CrawlBudget(11, 2, 12, 1, 18), resume=True)
    monkeypatch.setattr(Graph, 'save', original_save)
    with Browser(site, log, identity_mode='webnavigator', settle_ms=20) as browser:
        graph = crawl(browser, directory, log, budget, resume=True)
    events = [json.loads(s) for s in log.path.read_text(encoding='utf-8').splitlines()]
    pairs = Counter((e['source'], e['action']['id']) for e in events
                    if e['event'] == 'exploration_action_started')
    assert max(pairs.values()) == 1
    assert any(e['event'] == 'crawl_resumed' and e['skipped_attempted_actions'] > 0 for e in events)
    assert events[-1]['event'] == 'crawl_finished'
    assert len(graph.nodes) >= 3
    assert graph.data['attempts'] <= budget.max_attempts
    assert all(path.read_bytes() == content for path, content in saved_images.items())
    with pytest.raises(ValueError, match='Completed'):
        crawl(browser, directory, log, budget, resume=True)
