"""Offline coverage of the original Agent's Verified stop prompt wiring."""
import json
import os

import pytest

from scripts import run_webnavigator_flash_hard as runner
from web2kg.evaluation import webnavigator
from web2kg.evaluation.verified_answer import STOP_SPEC, stop_instruction, validate_stop
from test_verified_stop_retry import client, VALID


def test_vendored_prompt_and_examples_match_public_contract():
    path = runner.ROOT / 'third_party/webnavigator' / runner.STOP_PROMPT
    assert path.read_text(encoding='utf-8').strip() == STOP_SPEC
    assert 'Keep the original response sections' in STOP_SPEC
    assert '5h 47min' not in STOP_SPEC
    examples = []
    decoder = json.JSONDecoder()
    for line in STOP_SPEC.splitlines():
        if 'stop [{' not in line:
            continue
        answer, _ = decoder.raw_decode(line.split('stop [', 1)[1])
        validate_stop('ACTION: stop [' + json.dumps(answer) + ']')
        assert answer['task_type'] in {'MUTATE', 'RETRIEVE', 'NAVIGATE'}
        if answer['task_type'] == 'RETRIEVE':
            assert isinstance(answer['retrieved_data'], list)
        examples.append(answer)
    assert len(examples) == 3


def test_final_sent_prompt_has_one_contract_and_keeps_actor_sections(tmp_path):
    c, sent = client(tmp_path, [VALID])
    c.validate_verified_stop = False  # prompt-only adaptation, no new regeneration policy
    c.unified_verified_stop = True
    prompt = 'REASON and ACTION sections remain required.\n- ' + STOP_SPEC + '\n- click [id]: original click'
    assert c.call([{'role':'user','content':'unchanged observation'}], system_prompt=prompt) == VALID
    actual = sent[0]['instructions']
    assert actual == prompt == stop_instruction(prompt)
    assert actual.count('stop [answer]:') == 1
    assert 'When using stop [answer]' not in actual  # no contradictory appended legacy contract
    assert sent[0]['text']['format'] == {'type':'text'}
    assert len(sent) == 1


@pytest.mark.parametrize('enabled', [None, True])
def test_original_entry_passes_prompt_switch_without_enabling_other_changes(tmp_path, monkeypatch, enabled):
    seen = {}
    class ConstructorReached(Exception): pass
    def capture(*args, **kwargs):
        seen.update(kwargs)
        raise ConstructorReached
    monkeypatch.setattr(webnavigator, 'OriginalResponses', capture)
    config = dict(workspace=str(tmp_path), max_calls=31, output_tokens_per_call=4096, model='fixture')
    if enabled is not None:
        config['unified_verified_stop'] = enabled
    with pytest.raises(ConstructorReached):
        webnavigator.run_original(tmp_path, config, {'task_id':1})
    assert seen['unified_verified_stop'] is bool(enabled)
    assert not seen.get('validate_verified_stop')
    assert 'max_retries' not in seen


def test_new_registration_installs_only_stop_without_touching_hardlinked_source(tmp_path):
    runtime, batch = tmp_path/'runtime', tmp_path/'batch'
    source = runtime/'third_party/webnavigator'/runner.STOP_PROMPT
    source.parent.mkdir(parents=True)
    source.write_text(STOP_SPEC+'\n', encoding='utf-8')
    target = batch/'upstream'/runner.STOP_PROMPT
    target.parent.mkdir(parents=True)
    historical = tmp_path/'old-stop.txt'
    historical.write_text('historical prompt', encoding='utf-8')
    os.link(historical, target)
    other = target.parent/'type.txt'
    other.write_bytes(b'user-owned bytes')
    record = runner.install_stop_prompt(batch, runtime)
    assert target.read_bytes() == source.read_bytes()
    assert historical.read_text() == 'historical prompt'
    assert other.read_bytes() == b'user-owned bytes'
    assert record['sha256'] != record['previous_sha256']
    runner.check_continuation_prompt({'unified_verified_stop':True, 'stop_prompt_sha256':record['sha256']}, record['sha256'])
    with pytest.raises(ValueError, match='new experiment'):
        runner.check_continuation_prompt({}, record['sha256'])
    with pytest.raises(ValueError, match='new experiment'):
        runner.check_continuation_prompt({'unified_verified_stop':True, 'stop_prompt_sha256':'old'}, record['sha256'])
