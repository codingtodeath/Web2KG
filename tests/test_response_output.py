import json
from types import SimpleNamespace

import pytest

from web2kg.common.response_output import OutputError, parse_messages, parse_object_text
from web2kg.common.responses import ResponsesClient, ResponsesError, object_schema
from web2kg.common.storage import EventLog
# Historical JSON parser coverage; no longer the execution action protocol.
ACTION_SCHEMA = {"type": "object", "properties": {
    "action": {"type": "string", "enum": ["stop"]}, "element_id": {"type": ["string", "null"]},
    "value": {"type": "string"}, "reason": {"type": "string"}, "answer": {"type": "string"}},
    "required": ["action", "element_id", "value", "reason", "answer"], "additionalProperties": False}


def message(text, **metadata):
    return {"content": [{"type": "output_text", "text": text}], **metadata}


SCHEMA = object_schema({"ok": {"type": "boolean"}})


def test_message_boundaries_and_explicit_final_phase():
    messages = [message("Working on it", phase="commentary"),
                {"phase": "final_answer", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": '{"ok":'},
                             {"type": "output_text", "text": "true}"}]}]
    value, details = parse_messages(messages, SCHEMA)
    assert value == {"ok": True}
    assert details["selected_message_indices"] == [1]
    assert details["ignored_commentary_messages"] == 1
    with pytest.raises(OutputError, match="invalid_json"):
        parse_messages([message('{"ok":'), message("true}")], SCHEMA)


def test_identical_objects_return_once_with_diagnosis():
    value, details = parse_messages([message('{"ok":true}\n{"ok": true}'),
                                     message('{"ok":true}')], SCHEMA)
    assert value == {"ok": True}
    assert details["object_count"] == 3
    assert details["duplicate_objects_removed"] == 2


def test_explicit_fence_compatibility_preserves_schema_and_raw_messages():
    messages = [message(' \n```json\n{"ok":true}\n```\n')]
    value, details = parse_messages(messages, SCHEMA, allow_json_fence=True)
    assert value == {'ok': True} and details['json_fence_removed_messages'] == [0]
    assert '```' in messages[0]['content'][0]['text']
    with pytest.raises(OutputError, match='invalid_json'):
        parse_messages(messages, SCHEMA)


@pytest.mark.parametrize('text', [
    'Explanation\n```json\n{"ok":true}\n```',
    '```json\n{"ok":true}\n```\nExtra text',
    '```json\n{"ok":true}\n```\n```json\n{"ok":true}\n```',
    '```json\n{"ok":true}{"ok":false}\n```',
    '```json\n{"ok":true}{"ok":true}\n```',
    '```json\n{"ok":true,"ok":false}\n```',
    '```json\n{"ok":"true"}\n```',
    '```json\n{"ok":true}\n',
])
def test_fence_compatibility_does_not_extract_or_repair(text):
    with pytest.raises(OutputError):
        parse_messages([message(text)], SCHEMA, allow_json_fence=True)


@pytest.mark.parametrize("messages,reason", [
    ([message('{"ok":true}{"ok":false}')], "conflicting_objects"),
    ([message('{"ok":true}'), message('{"ok":false}')], "conflicting_messages"),
    ([message('{"ok":false}'), message('{"ok":true}', phase="final_answer")], "conflicting_messages"),
    ([message('{"ok":true}', status="incomplete")], "incomplete_message"),
    ([message('{"ok":true}', role="user")], "unexpected_role"),
    ([message('{"ok":true}', phase="commentary")], "missing_or_unknown_final_phase"),
    ([{"content": [{"type": "refusal", "refusal": "No"}]}], "refusal"),
    ([message('{"ok":"true"}')], "schema_mismatch"),
    ([message('{"ok":true,"extra":1}')], "schema_mismatch"),
    ([message('{}')], "schema_mismatch"),
    ([], "missing_message"),
])
def test_ambiguous_or_invalid_output_rejected(messages, reason):
    with pytest.raises(OutputError, match=reason):
        parse_messages(messages, SCHEMA)


@pytest.mark.parametrize("text,reason", [
    ('{"ok":true} trailing', "invalid_json"),
    ('{"ok":true}{"ok":', "invalid_json"),
    ('```json\n{"ok":true}\n```', "invalid_json"),
    ('[{"ok":true}]', "expected_object"),
    ('{"ok":true,"ok":false}', "duplicate_json_key"),
    ('{"ok":NaN}', "nonfinite_json_number"),
    ('{"ok":true}{"ok":1}', "conflicting_objects"),
    (' ', "empty_output"),
])
def test_no_guessing_or_partial_json_repair(text, reason):
    with pytest.raises(OutputError, match=reason):
        parse_object_text(text)


def test_nested_answer_string_is_not_an_extra_action():
    action = {"action": "stop", "element_id": None, "value": "", "reason": "done",
              "answer": json.dumps({"retrieved_data": [{"text": 'quotes " and }{' }]})}
    value, details = parse_messages([message(json.dumps(action))], ACTION_SCHEMA)
    assert value == action and details["object_count"] == 1
    action["action"] = "invented"
    with pytest.raises(OutputError, match="schema_mismatch"):
        parse_messages([message(json.dumps(action))], ACTION_SCHEMA)


@pytest.mark.parametrize("text,status,accepted", [
    ('{"ok":true}{"ok":true}', "completed", True),
    ('{"ok":true}{"ok":false}', "completed", False),
    ('{"ok":true}', "incomplete", False),
    ('{"ok":"true"}', "completed", False),
])
def test_client_persists_boundaries_usage_and_budget_without_retry(tmp_path, text, status, accepted):
    env = tmp_path / "private.env"
    env.write_text("CRAWL_API_KEY=test-secret\nCRAWL_BASE_URL=https://example.org/v1\nCRAWL_MODEL=test\n")
    log = EventLog(tmp_path / "run")
    client = ResponsesClient(env, log, max_calls=1)
    calls = []
    def post(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(ok=True, json=lambda: {"status": status, "model": "test",
            "usage": {"output_tokens": 12}, "output": [
                {"type": "reasoning", "encrypted_content": "not-for-log"},
                {"type": "message", "id": "m1", **message(text)}]})
    client.session.post = post
    if accepted:
        assert client.json("JSON", {}, SCHEMA, "test") == {"ok": True}
    else:
        with pytest.raises(ResponsesError):
            client.json("JSON", {}, SCHEMA, "test")
    with pytest.raises(ResponsesError, match="budget"):
        client.json("JSON", {}, SCHEMA, "test")
    assert calls == [1]
    saved = json.loads(next((log.directory / "llm").glob("*/response.json")).read_text())
    assert saved["messages"][0]["id"] == "m1"
    assert saved["usage"] == {"output_tokens": 12}
    assert saved["parse"]["status"] == ("accepted" if accepted else "rejected")
    assert "not-for-log" not in json.dumps(saved) and "test-secret" not in log.path.read_text()
