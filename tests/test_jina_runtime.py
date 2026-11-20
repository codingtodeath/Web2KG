import json
from unittest.mock import Mock

import pytest

from web2kg.common.jina_runtime import LocalEncoder
from web2kg.common.storage import Graph


def runtime(tmp_path, monkeypatch, identity):
    graph = Graph(tmp_path / 'graph')
    graph.directory.mkdir()
    (graph.directory / 'index.json').write_text(json.dumps({'mode': 'jina_screenshot',
        'encoder': identity, 'endpoint': 'http://127.0.0.1:8793/v1/embeddings'}))
    client = Mock()
    monkeypatch.setattr('web2kg.common.jina_runtime.JinaClient', lambda _: client)
    python = tmp_path / 'python.exe'
    python.touch()
    return LocalEncoder(graph, tmp_path, python), client


def test_existing_encoder_is_checked_and_never_stopped(tmp_path, monkeypatch):
    service, client = runtime(tmp_path, monkeypatch, {'fingerprint': 'saved'})
    client.identity.return_value = {'fingerprint': 'saved'}
    launch = Mock()
    monkeypatch.setattr('web2kg.common.jina_runtime.subprocess.Popen', launch)
    service.ensure()
    service.close()
    launch.assert_not_called()
    client.identity.return_value = {'fingerprint': 'different'}
    with pytest.raises(ValueError, match='不一致'):
        service.ensure()


def test_owned_encoder_uses_index_precision_and_is_closed(tmp_path, monkeypatch):
    identity = {'fingerprint': 'saved', 'pipeline': 'retrieval-vision-fp32-v1'}
    service, client = runtime(tmp_path, monkeypatch, identity)
    client.identity.side_effect = [ConnectionError(), identity]
    process = Mock()
    process.poll.return_value = None
    launch = Mock(return_value=process)
    monkeypatch.setattr('web2kg.common.jina_runtime.subprocess.Popen', launch)
    service.ensure()
    args, kwargs = launch.call_args
    assert '--vision-fp32' in args[0]
    assert kwargs['env']['HF_HUB_OFFLINE'] == '1'
    service.close()
    process.terminate.assert_called_once()
    assert service.stream.closed
