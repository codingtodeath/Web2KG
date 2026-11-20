"""Run the registered original WebNavigator execution baseline, without replaying started tasks."""
import argparse
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_verified_template import check_inputs, queue
from web2kg.common.activity import read_events
from web2kg.common.storage import write_json
from web2kg.evaluation.webnavigator import run_original


def summarize(batch, config, state):
    comparison = config.get('comparison_summary')
    baseline = json.loads(Path(comparison).read_text(encoding='utf-8')) if comparison else {'tasks': []}
    previous = {r['task_id']: r['official_score'] for r in baseline['tasks']}
    rows = []
    for key in config['task_ids']:
        folder = batch / str(key)
        def read(name):
            path = folder / name
            return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        result, official = read('result.json'), read('official.json')
        events = read_events(folder / 'events.jsonl')
        responses = [e for e in events if e['event'] == 'responses_received']
        navigation = [e for e in events if e['event'] == 'original_navigation']
        rows.append(dict(task_id=key, status=state.get(str(key), {}).get('status', 'not_started'),
            agent_status=result.get('status'), official_score=official.get('score'), template_score=previous.get(key),
            steps=result.get('steps'), seconds=result.get('seconds'), error=result.get('error'),
            atomic_actions=result.get('atomic_actions'),
            calls=sum(e['event'] == 'responses_started' for e in events), responses=len(responses),
            actor_calls=sum(e['event'] == 'responses_started' and e.get('purpose') == 'original_actor' for e in events),
            selector_calls=sum(e['event'] == 'responses_started' and e.get('purpose') == 'page_selector' for e in events),
            input_tokens=sum(e.get('usage', {}).get('input_tokens', 0) for e in responses),
            output_tokens=sum(e.get('usage', {}).get('output_tokens', 0) for e in responses),
            unknown_usage=sum(not e.get('usage') for e in responses),
            actual_models=sorted({str(e.get('actual_model')) for e in responses}),
            navigation_attempts=len(navigation), navigation_returned_true=sum(e.get('returned') is True for e in navigation)))
    pairs = [r for r in rows if r['official_score'] is not None and r['template_score'] is not None]
    summary = dict(planned=len(rows), official_scored=sum(r['official_score'] is not None for r in rows),
        paired_scored=len(pairs), comparison_summary=comparison,
        official_successes=sum(r['official_score'] == 1 for r in rows), tasks=rows,
        original_only=sum(r['official_score'] == 1 and r['template_score'] == 0 for r in pairs),
        template_only=sum(r['official_score'] == 0 and r['template_score'] == 1 for r in pairs),
        scope='Whole-system comparison on previously exposed tasks; graphs, observations, Agent and navigation differ.')
    write_json(batch / 'summary.json', summary)
    intro = ('原仓库 Agent/提示/截图选择/导航；Terra Responses、独立 Verified 评分。'
        '原发布图225节点及201张图的原始编码，文本查询适配本地Jina；top-k5，30步/31调用。'
        '原代码允许动作格式重新生成，但所有请求计入同一31次上限；HTTP失败无重试。'
        '对照分数仅来自协议指定的comparison_summary，未指定时留空；两套图谱、观察、动作与导航不同，不能单独归因模板。导航返回true不等于任务成功。')
    header = ['任务', '模板分', 'WebNavigator分', '状态', '步数', '调用', '导航返回true/尝试']
    table = '| ' + ' | '.join(header) + ' |\n|' + '|'.join(['---']*len(header)) + '|\n'
    for r in rows:
        table += '| ' + ' | '.join(str(v) for v in [r['task_id'], r['template_score'], r['official_score'],
            r['agent_status'], r['steps'], r['calls'], f"{r['navigation_returned_true']}/{r['navigation_attempts']}"]) + ' |\n'
    (batch / 'report.md').write_text('# 原仓库 WebNavigator × Reddit Hard 10\n\n' + intro + '\n\n' + table, encoding='utf-8')
    page = '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>WebNavigator 对照</title><style>body{font:16px/1.7 system-ui;max-width:1200px;margin:30px auto}pre{white-space:pre-wrap}</style><h1>WebNavigator 对照</h1><p>' + intro + '</p><pre>' + html.escape(table) + '</pre>'
    (batch / 'report.html').write_text(page, encoding='utf-8')
    (Path(config['graph']) / 'reports' / (batch.name + '.html')).write_text(page, encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--worker', type=int)
    args = parser.parse_args(); batch = args.directory.resolve()
    config = json.loads((batch / 'protocol.json').read_text(encoding='utf-8'))
    if args.worker is not None:
        check_inputs(config)
        tasks = json.loads((batch / 'public-inputs.json').read_text(encoding='utf-8'))
        task = next(t for t in tasks if t['task_id'] == args.worker)
        if set(task) != {'task_id', 'sites', 'intent', 'start_urls'}:
            raise ValueError('Non-public task fields')
        run_original(batch, config, task)
    else:
        queue(batch, config, worker_script=Path(__file__).resolve(), summarizer=summarize)


if __name__ == '__main__':
    main()
