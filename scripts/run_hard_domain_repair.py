"""Register and run only the authorized615/617/618 navigation-domain repairs."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from scripts import run_reddit_hard10_225 as run
from web2kg.stage1.site_framework import load_framework
from web2kg.stage2.template_context import TemplateKnowledge

PREVIOUS = {
    615: 'reddit-hard22-feedback-flash-20261106-v1',
    617: 'reddit-hard22-feedback-flash-20261106-v1',
    618: 'reddit-hard8-cardfix-flash-20261106-v1',
}


def retry_inputs(task):
    previous = ROOT/'runs'/PREVIOUS[task]
    folder = previous/str(task)
    old = run.read(folder/'input.json')
    result = run.read(folder/'result.json')
    if result.get('error') != 'invalid_domain_max_retries_exceeded':
        raise ValueError('This authorization covers the navigation-domain failure only')
    graph = Path(old['graph'])
    # Use each task's own original input, not another task's newly learned graph.
    for path, expected in run.read(graph.parent/'knowledge-handoff.json').items():
        if run.file_hash(path) != expected:
            raise ValueError('Original task input/experience changed: '+path)
    if graph == folder/'execution-graph':
        raise ValueError('Never seed from the failed attempt')
    return previous, folder, old, graph


def register(batch, checks_path):
    if batch.exists() or not batch.is_relative_to(ROOT/'runs'):
        raise ValueError('Use a fresh directory inside runs')
    checks = run.read(checks_path)
    if not checks.get('passed'):
        raise ValueError('Relevant offline checks must pass')
    for rel, expected in checks['source_sha256'].items():
        if run.file_hash(ROOT/rel) != expected:
            raise ValueError('Tested source changed: '+rel)
    protected = {}
    for name in sorted(set(PREVIOUS.values())):
        previous = ROOT/'runs'/name
        run.verify(previous, record=False)
        protected.update(run.read(previous/'protected-inputs.json'))
        protected.update(run.read(run.read(previous/'state.json')['knowledge_manifest']))
        for rel, sha in run.read(previous/'frozen-runtime/manifest.json').items():
            protected[str(previous/'frozen-runtime'/rel)] = sha
        for name in ('state.json', 'summary.json', 'combined-audit.json', 'report.html', 'analysis.html'):
            path = previous/name
            if path.exists(): protected[str(path)] = run.file_hash(path)
    sources = {task: retry_inputs(task) for task in PREVIOUS}
    batch.mkdir()
    runtime = run.freeze(batch)
    protocol = dict(version='navigation-domain-feedback-v1', runtime=str(runtime), tasks=list(PREVIOUS),
        created_at=datetime.now(timezone.utc).isoformat(),
        authorization='User authorized fixing navigation argument feedback and one retry each of615,617,618 only.',
        knowledge_policy='Independent original per-task225-node input and verified original experience. No cross-retry learning.',
        stop_policy='First engineering/permanent/asset error stops all remaining cases. No automatic restart.',
        paid_preflight=0, checks=str(checks_path),
        output_policy='Same observation; latest rejected response and actual registered site IDs; maximum3 invalid-domain responses, within shared3-regeneration budget. No browser action or extra logical decision for rejected output; physical usage retained.',
        limits='Three exposed development failures, not a new full-suite score or causal/generalization comparison. Original41 results and all failed-attempt costs remain separate.')
    run.write_json(batch/'protocol.json', protocol)
    run.write_json(batch/'state.json', dict(status='registered', tasks={}))
    for task, (previous, original, old, graph) in sources.items():
        child = batch/f'case-{task}'; child.mkdir()
        seed = child/'seed-graph'; shutil.copytree(graph, seed)
        memory = child/'initial-experience.json'; shutil.copy2(old['experience_store'], memory)
        encoder = child/'encoder-runtime/scripts'; encoder.mkdir(parents=True)
        for name in ('serve_jina_gpu.py', 'jina_gpu.py', 'jina_vision_fp32.py'):
            shutil.copy2(Path(old['encoder_runtime'])/'scripts'/name, encoder/name)
        g = run.Graph.load(seed); index = run.read(seed/'index.json')
        TemplateKnowledge(g, run.cards_path(seed), run.EventLog(child/'registration-events'))
        if len(g.nodes) != 225 or set(index['ids']) != set(g.nodes) or not load_framework(g):
            raise ValueError('Require the complete225-node original task input')
        if index['encoder'] != old['encoder_expected']:
            raise ValueError('Encoder identity mismatch')
        config = {**old, 'version':'hard3-225-domain-feedback-v1', 'created_at':protocol['created_at'],
            'runtime':str(runtime), 'task_ids':[task], 'inherited_results':[],
            'authorization':protocol['authorization'], 'previous_version':str(previous),
            'source_graph':str(graph), 'initial_graph':str(seed), 'prepared_graph':str(seed),
            'graph':str(seed), 'cards':str(run.cards_path(seed)),
            'initial_experience':str(memory), 'experience_store':str(child/'experience.json'),
            'stop_file':str(batch/'STOP'), 'encoder_runtime':str(encoder.parent),
            'output_retry_feedback':'previous-response-v1', 'navigation_domain_validation':'registered-site-feedback-v1',
            'card_output_validation':'validated-card-v1',
            'knowledge_policy':protocol['knowledge_policy'], 'restart_policy':protocol['stop_policy'],
            'new_card_preparation':'Original complete input; no paid preflight or preparation.',
            'report_title':f'Reddit Hard{task} · 导航参数纠错验证', 'limits':protocol['limits'],
            'excluded_attempts':[dict(task_id=task, directory=str(original),
                state=run.read(previous/'state.json')['tasks'][str(task)],
                reason='用户授权修复导航参数反馈后重跑；原失败、评分与成本保留，不计入本次三题结果')]}
        run.write_json(child/'protocol.json', config)
        public = [r for r in run.read(previous/'public-inputs.json') if r['task_id'] == task]
        run.write_json(child/'public-inputs.json', public)
        run.write_json(child/'repair-lineage.json', dict(original_attempt=str(original),
            original_graph=str(graph), original_input_sha256=run.file_hash(original/'input.json'),
            original_experience_sha256=run.file_hash(old['experience_store']),
            current_source_baseline='latest card-validation repair plus navigation-domain feedback only; other diagnosed problems unchanged'))
        run.write_json(child/'run_material_passport.json', dict(mode='run', new_starts=1,
            graph_nodes=225, models='All Qwen3.7-flash; Bailian direct', paid_preflight=0, limits=protocol['limits']))
        inputs = {**run.hashes(seed), **run.hashes(encoder.parent)}
        for path in [batch/'protocol.json', memory, checks_path,
                     *[child/n for n in ('protocol.json','public-inputs.json','repair-lineage.json','run_material_passport.json')]]:
            inputs[str(path)] = run.file_hash(path)
        run.write_json(child/'runtime-inputs.json', inputs)
        guard = {**protected, **run.hashes(graph), **run.hashes(original), **inputs,
                 str(child/'runtime-inputs.json'):run.file_hash(child/'runtime-inputs.json')}
        run.write_json(child/'protected-inputs.json', guard)
        run.write_json(child/'state.json', dict(status='registered', tasks={}, current_graph=str(seed)))
        run.verify(child); run.report(child)
    print(dict(registered=str(batch), tasks=list(PREVIOUS), independent_original_inputs=True))


def live(batch):
    protocol = run.read(batch/'protocol.json'); state = run.read(batch/'state.json')
    if Path(protocol['runtime']) != ROOT or protocol['tasks'] != list(PREVIOUS):
        raise ValueError('Use this registration frozen entry and fixed three tasks')
    if state['status'] != 'registered' or state['tasks']:
        raise ValueError('Never restart a started validation')
    with run.queue_lock(batch/'queue.lock'):
        state.update(status='running', started_at=time.time())
        run.write_json(batch/'state.json', state)
        try:
            for task in protocol['tasks']:
                if (batch/'STOP').exists():
                    state['status']='user_stopped'; break
                child = batch/f'case-{task}'
                state['tasks'][str(task)]='started'; run.write_json(batch/'state.json', state)
                run.queue(child)
                child_state = run.read(child/'state.json')
                state['tasks'][str(task)] = child_state['status']
                run.write_json(batch/'state.json', state)
                if child_state['status'] != 'completed':
                    state['status'] = child_state['status']; break
            else:
                state['status']='completed'
        except Exception as error:
            state.update(status='needs_attention', error=type(error).__name__+': '+str(error))
            raise
        finally:
            state['seconds']=time.time()-state['started_at']
            run.write_json(batch/'state.json', state)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--checks', type=Path)
    p.add_argument('--live', action='store_true')
    a=p.parse_args()
    if a.live: live(a.directory.resolve())
    else: register(a.directory.resolve(), a.checks.resolve())
