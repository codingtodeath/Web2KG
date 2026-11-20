import ast
from pathlib import Path
import re

import pytest

from web2kg.evaluation.webnavigator import parse_heading_elements


@pytest.mark.parametrize('colon', [':', '：'])
@pytest.mark.parametrize('newline', ['\n', '\r\n'])
def test_reason_alias_and_action_are_distinct(colon, newline):
    text = newline.join([f'REASON FOR ACTION{colon}', 'Open the post.',
                         f'ACTION{colon}', 'click [4722]',
                         f'OBSERVATION HIGHLIGHT{colon}', '4722'])
    assert parse_heading_elements(None, text, ['reason','action','observation highlight']) == {
        'reason':'Open the post.', 'action':'click [4722]', 'observation highlight':'4722'}


def test_no_action_is_invented_from_reason_or_inline_text():
    assert parse_heading_elements(None, 'REASON FOR ACTION:\nclick [1]', ['action']) == {'action':''}
    assert parse_heading_elements(None, 'This mentions ACTION: click [1]', ['action']) == {'action':''}


def test_regular_headings_multiline_and_missing_fields():
    text = 'REASON: First line\nSecond line\n\tACTION: go_back\nOBSERVATION HIGHLIGHT: 3'
    assert parse_heading_elements(None, text, ['reason','action','missing']) == {
        'reason':'First line\nSecond line','action':'go_back','missing':''}


def test_does_not_silently_change_multiple_command_policy():
    text = 'ACTION:\nclick [1]\ngo_back'
    assert parse_heading_elements(None, text, ['action'])['action'] == 'click [1]\ngo_back'


def test_real_failed_outputs_preserve_commands_and_reason():
    root = Path(__file__).resolve().parents[1]
    batch = root/'runs/webnavigator-flash-hard42-20261108-v1'
    if not batch.exists(): pytest.skip('Local preserved failure corpus unavailable')
    import json
    diagnosis = json.loads((batch/'failure-diagnosis.json').read_text('utf-8'))
    source = batch/'upstream/webNavigator/webNavigator.py'
    node = next(n for n in ast.walk(ast.parse(source.read_text('utf-8')))
                if isinstance(n, ast.FunctionDef) and n.name == 'parse_elements')
    scope = {'re':re}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    assert diagnosis['misparsed_count'] == 15
    for row in diagnosis['misparsed_responses']:
        text = (batch/'598/llm'/row['request_id']/'parsed.txt').read_text('utf-8')
        fixed = parse_heading_elements(None, text, ['reason','action'])
        assert fixed['action'] == row['explicit_action']
        assert fixed['reason']
        assert scope['parse_elements'](None,text,['action'])['action'] != fixed['action']
