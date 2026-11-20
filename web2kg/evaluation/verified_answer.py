"""Validate only the public stop-answer format, never task correctness."""
import json
import re

from ..common.responses import ResponsesError


STOP_SPEC = '''stop [answer]: Finish the task and return ONE JSON object inside the action brackets.
Keep the original response sections and ACTION command syntax. Only the answer inside stop [...] is JSON; do not replace the whole response with JSON.
Write actual JSON syntax with double-quoted keys/strings, without Markdown fences. The object has:
- task_type (required): "RETRIEVE", "MUTATE" or "NAVIGATE" according to the user's task.
- status (required): "SUCCESS", "ACTION_NOT_ALLOWED_ERROR", "NOT_FOUND_ERROR", "PERMISSION_DENIED_ERROR", "DATA_VALIDATION_ERROR" or "UNKNOWN_ERROR".
- retrieved_data: an array of the requested items for RETRIEVE, including when there is only one item; null or omitted for other tasks. Preserve the requested item keys and value types.
- error_details: null or omitted on success; otherwise a short explanation of what could not be completed or verified.
Choose status from observed evidence; stopping does not imply success. Do not infer absence from one empty search.
Example command ONLY when a requested change is already visibly completed: stop [{"task_type":"MUTATE","status":"SUCCESS","retrieved_data":null}]
Retrieval format example: stop [{"task_type":"RETRIEVE","status":"SUCCESS","retrieved_data":["observed item"]}] Replace the example item with the actual requested data from observations.
Alternative example when the requested change cannot be verified: stop [{"task_type":"MUTATE","status":"UNKNOWN_ERROR","error_details":"The requested change could not be verified."}]
These are alternatives, not a sequence. Never append stop to a click/type action in the same response.
Even when the task requests no textual answer, return the object with task_type and status. Never return a bare sentence, SUCCESS, N/A, null, array, or unwrapped data object.
Use stop only for the final answer. Put intermediate memory in NOTE. No evaluator reference answers are supplied.'''


def stop_instruction(instructions):
    """Replace the legacy two-line command specification, not unrelated page text."""
    if STOP_SPEC in instructions:
        return instructions
    pattern = r'(?m)^[ \t]*(?:-[ \t]*)?stop \[answer\]:[^\n]*(?:\nSTRICT RULE:[^\n]*)?'
    updated, count = re.subn(pattern, lambda _: STOP_SPEC, instructions)
    return updated if count else instructions + '\n' + STOP_SPEC


def validate_stop(text, *, required=False):
    section = re.search(r'^ACTION:[ \t]*\n?(.*?)(?=\n[A-Z][A-Z \d]*:|\Z)', text, re.M | re.S)
    action = section[1].strip() if section else ''
    is_stop = bool(re.match(r'^stop\b', action))
    if not is_stop and not required:
        return
    try:
        match = re.fullmatch(r'stop\s*\[(.*)\]', action, re.S)
        if not match:
            raise ValueError('Expected one stop command')
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError('Duplicate JSON key')
                result[key] = value
            return result
        def constant(value):
            raise ValueError('Non-finite JSON number')
        answer = match[1].strip()
        fenced = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', answer, re.S)
        if fenced:
            answer = fenced[1].strip()
        value = json.loads(answer, object_pairs_hook=pairs, parse_constant=constant)
        if not isinstance(value, dict):
            raise ValueError('Expected a JSON object, not prose, null or a scalar')
        task_type = value.get('task_type', value.get('performed_operation'))
        if not isinstance(task_type, str) or task_type.strip().upper() not in {'RETRIEVE', 'MUTATE', 'NAVIGATE'}:
            raise ValueError('Invalid task_type')
        if not isinstance(value.get('status'), str) or value['status'].strip().upper() not in {
                'SUCCESS', 'ACTION_NOT_ALLOWED_ERROR', 'NOT_FOUND_ERROR', 'PERMISSION_DENIED_ERROR',
                'DATA_VALIDATION_ERROR', 'UNKNOWN_ERROR'}:
            raise ValueError('Invalid status')
        if value.get('error_details') is not None and not isinstance(value['error_details'], str):
            raise ValueError('error_details must be text or null')
    except (ValueError, TypeError) as error:
        raise ResponsesError('Invalid public stop format: ' + str(error), kind='output:verified_stop_invalid') from None
