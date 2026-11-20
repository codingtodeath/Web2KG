from scripts import register_hard42_60 as reg


def test_prior_ten_inherit_only_task_selection(tmp_path):
    ref = reg.run.read(reg.REFERENCE / 'protocol.json')
    rows = reg.task_set(ref, 'prior-hard10')
    cfg = reg.settings(tmp_path, ref, rows)
    assert cfg['task_ids'] == [646, 598, 625, 596, 721, 31, 730, 409, 651, 644]
    assert cfg['inherited_results'] == cfg['excluded_attempts'] == []
    assert 'initial_experience' not in cfg and 'previous_version' not in cfg
    assert cfg['prepared_graph'] == str(tmp_path / 'initial-graph')
    assert cfg['max_graph_nodes'] == 60 and cfg['actor_prompt'] == 'single-action-v2'
    assert cfg['max_output_retries'] == 3 and cfg['validate_verified_stop'] and cfg['unified_verified_stop']
    assert '10 fresh starts' in cfg['restart_policy'] and 'Hard10' in cfg['report_title']


def test_full_task_set_uses_original60_and_empty_experience(tmp_path):
    ref = reg.run.read(reg.REFERENCE / 'protocol.json')
    rows = reg.run.selected_tasks(reg.ROOT, ref['logical_origin'], ref['seed'], limit=None)
    cfg = reg.settings(tmp_path, ref, rows)
    assert len(cfg['task_ids']) == len(set(cfg['task_ids'])) == 42
    assert cfg['max_graph_nodes'] == 60
    assert cfg['source_graph'] == str(reg.SOURCE)
    assert cfg['prepared_graph'] == cfg['initial_graph'] == str(tmp_path / 'initial-graph')
    assert cfg['inherited_results'] == cfg['excluded_attempts'] == []
    assert 'initial_experience' not in cfg and 'previous_version' not in cfg
    assert all(cfg[k] == 'qwen3.7-flash' for k in ('model', 'auxiliary_model', 'diagnosis_model'))
    assert cfg['integrity_policy'] == 'boundary-v1' and cfg['recovery_actions'] == 12


def test_report_uses_registered_capacity_and_includes_diagnosis(tmp_path, monkeypatch):
    run = reg.run
    run.write_json(tmp_path / 'protocol.json', dict(task_ids=[1], model='qwen3.7-flash',
        max_graph_nodes=60, limits='development', initial_graph='initial'))
    run.write_json(tmp_path / 'state.json', dict(status='running', tasks={}, current_graph='current'))
    run.write_json(tmp_path / '1/recovery/recovery.json', dict(decisions=[{'decision': {'root_step': 2}}],
        attempted=True, **{'return': {'success': True, 'route': 'root_entry', 'steps': 3}}))
    summary = run.report(tmp_path)
    assert summary['tasks'][0]['recovery']['return']['route'] == 'root_entry'
    assert summary['tasks'][0]['recovery']['decisions'][0]['decision']['root_step'] == 2
    assert '收尾容量60' in (tmp_path / 'report.html').read_text(encoding='utf-8')
