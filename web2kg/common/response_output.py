"""Extract one unambiguous structured result without merging independent messages."""
import json
import re

from jsonschema import Draft202012Validator


class OutputError(ValueError):
    def __init__(self, code, **details):
        super().__init__(code)
        self.code, self.details = code, details


def output_messages(data):
    # Preserve visible output boundaries, never hidden reasoning or provider headers.
    return [{**{key: item.get(key) for key in ("id", "role", "status", "phase")},
             "content": [{key: part[key] for key in ("type", "text", "refusal") if key in part}
                         for part in item.get("content", [])]}
            for item in data.get("output", []) if item.get("type") == "message"]


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise OutputError("duplicate_json_key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise OutputError("nonfinite_json_number")


def parse_object_text(text):
    """Allow repeated identical objects only; never choose among conflicting actions."""
    decoder = json.JSONDecoder(object_pairs_hook=_unique_keys, parse_constant=_invalid_constant)
    objects, position = [], 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        try:
            value, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as error:
            raise OutputError("invalid_json", position=error.pos) from None
        if not isinstance(value, dict):
            raise OutputError("expected_object")
        objects.append(value)
    if not objects:
        raise OutputError("empty_output")
    # Canonical JSON keeps booleans and numbers distinct, unlike Python dict equality.
    keys = {json.dumps(value, sort_keys=True, ensure_ascii=False) for value in objects}
    if len(keys) != 1:
        raise OutputError("conflicting_objects", object_count=len(objects))
    return objects[0], len(objects)


def parse_messages(messages, schema, *, allow_json_fence=False):
    if not messages:
        raise OutputError("missing_message")
    has_final = any(message.get("phase") == "final_answer" for message in messages)
    values, indices, count, fenced = [], [], 0, []
    for index, message in enumerate(messages):
        if message.get("role") not in (None, "assistant"):
            raise OutputError("unexpected_role", message_index=index)
        phase = message.get("phase")
        if phase == "commentary" and has_final:
            continue
        if phase not in (None, "final_answer"):
            raise OutputError("missing_or_unknown_final_phase", message_index=index)
        if message.get("status") not in (None, "completed"):
            raise OutputError("incomplete_message", message_index=index)
        parts = message.get("content", [])
        if any(part.get("type") == "refusal" for part in parts):
            raise OutputError("refusal", message_index=index)
        if any(part.get("type") != "output_text" for part in parts):
            raise OutputError("unexpected_content", message_index=index)
        # Text parts can fragment a single message. Independent messages cannot.
        raw = "".join(part["text"] for part in parts)
        if allow_json_fence:
            wrapper = re.fullmatch(r'\s*```(?:json)?[ \t]*\r?\n(.*?)\r?\n```\s*', raw, re.DOTALL | re.IGNORECASE)
            if wrapper and '```' not in wrapper[1]:
                raw = wrapper[1]
                fenced.append(index)
        value, object_count = parse_object_text(raw)
        if index in fenced and object_count != 1:
            raise OutputError('multiple_objects_in_fence', message_index=index)
        error = next(Draft202012Validator(schema).iter_errors(value), None)
        if error is not None:
            raise OutputError("schema_mismatch", message_index=index,
                              field_path=list(error.absolute_path), rule=error.validator)
        values.append(value)
        indices.append(index)
        count += object_count
    keys = {json.dumps(value, sort_keys=True, ensure_ascii=False) for value in values}
    if len(keys) != 1:
        raise OutputError("conflicting_messages", message_count=len(values))
    return values[0], {"selected_message_indices": indices, "object_count": count,
                       "json_fence_removed_messages": fenced,
                       "duplicate_objects_removed": count - 1,
                       "ignored_commentary_messages": len(messages) - len(indices)}
