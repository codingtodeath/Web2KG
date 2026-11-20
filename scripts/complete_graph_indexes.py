"""Complete both current Reddit indexes in independent copies, with one owned local encoder."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web2kg.common.storage import Graph, EventLog, write_json
from web2kg.common.jina import JinaClient, sha256
from web2kg.stage1.incremental_index import extend_visual_index
from complete_paused_graph_index import CheckpointClient


def main():
    project, out = map(lambda p: Path(p).resolve(), sys.argv[1:3])
    if not out.is_relative_to(project / 'runs') or out.exists():
        raise ValueError('Choose a fresh workspace runs directory')
    priors = ['reddit-jev-flash-expand225-20261104-v2', 'reddit-jev-flash-expand60-20261104-v1']
    out.mkdir()
    shutil.copytree(project / 'web2kg', out / 'source/web2kg', ignore=shutil.ignore_patterns('__pycache__'))
    scripts = out / 'source/scripts'
    scripts.mkdir()
    for name in ('complete_graph_indexes.py', 'complete_paused_graph_index.py'):
        shutil.copy2(project / 'scripts' / name, scripts / name)
    for name in ('serve_jina_gpu.py', 'jina_gpu.py', 'jina_vision_fp32.py'):
        shutil.copy2(project / 'runs' / priors[0] / 'source/scripts' / name, scripts / name)
    inputs = []
    for name in priors:
        base = project / 'runs' / name / 'graph'
        data = json.loads((base / 'index.json').read_text(encoding='utf-8'))
        inputs.append({'prior': name, 'graph_sha256': sha256(base / 'graph.json'),
                       'index_sha256': sha256(base / 'index.json'),
                       'vectors_sha256': sha256(base / data['vectors']), 'encoder': data['encoder']})
        shutil.copytree(base, out / name / 'graph')
    assert inputs[0]['encoder'] == inputs[1]['encoder']
    write_json(out / 'plan.json', {'inputs': inputs, 'llm_calls': 0, 'browser_actions': 0,
        'publish_active_graph': False, 'node_scope': 'all_graph_nodes', 'port': 8797})
    # Run only the copied implementation; development edits cannot change this job.
    result = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(scripts / 'complete_graph_indexes.py'),
                             '--worker', str(project), str(out)], cwd=project)
    raise SystemExit(result.returncode)


def worker(project, out):
    import numpy as np
    plan = json.loads((out / 'plan.json').read_text(encoding='utf-8'))
    port = plan['port']
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
    log = EventLog(out / 'index-calls')
    client = JinaClient(f'http://127.0.0.1:{port}/v1/embeddings', log, timeout=1800,
                        retry_connections=True, stop_file=out / 'STOP')
    state = {'status': 'running', 'graphs': []}
    write_json(out / 'state.json', state)
    started = time.monotonic()
    process = None
    with (out / 'encoder.log').open('x', encoding='utf-8') as stream:
        try:
            process = subprocess.Popen(['D:/Anaconda/envs/web2kg-jina-gpu/python.exe', '-B',
                str(ROOT / 'scripts/serve_jina_gpu.py'), '--port', str(port), '--vision-fp32',
                '--gpu-memory', '4GiB', '--log', str(out / 'encoder-events.jsonl')],
                cwd=project, env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                                 'PYTHONDONTWRITEBYTECODE': '1'},
                stdout=stream, stderr=stream, creationflags=subprocess.CREATE_NO_WINDOW)
            write_json(out / 'owned-process.json', {'pid': process.pid, 'parent': os.getpid(), 'port': port})
            while True:
                if (out / 'STOP').exists():
                    raise RuntimeError('User stop')
                if process.poll() is not None:
                    raise RuntimeError('Owned encoder exited')
                try:
                    identity = client.identity()
                    break
                except OSError:
                    time.sleep(1)
            if identity != plan['inputs'][0]['encoder']:
                raise ValueError('Encoder identity differs; no vector mixing')
            checkpoint = CheckpointClient(client, out / 'vector-checkpoints')
            for row in plan['inputs']:
                base = Graph.load(project / 'runs' / row['prior'] / 'graph')
                graph = Graph.load(out / row['prior'] / 'graph')
                index = extend_visual_index(base, graph, checkpoint, log)
                with np.load(graph.directory / index['vectors'], allow_pickle=False) as vectors:
                    assert set(vectors.files) == set(index['ids']) == set(graph.nodes)
                assert sha256(base.directory / 'graph.json') == row['graph_sha256']
                assert sha256(base.directory / 'index.json') == row['index_sha256']
                assert sha256(graph.directory / 'graph.json') == row['graph_sha256']
                state['graphs'].append({'prior': row['prior'], 'directory': str(graph.directory),
                    'pages': len(index['ids']), 'reused': len(index['incremental']['reused_ids']),
                    'filled': len(index['incremental']['encoded_ids'])})
                write_json(out / 'state.json', state)
            state['status'] = 'completed'
        except Exception as error:
            state.update(status='failed', error_type=type(error).__name__)
            raise
        finally:
            if process and process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            state['seconds'] = time.monotonic() - started
            write_json(out / 'state.json', state)


if __name__ == '__main__':
    if sys.argv[1] == '--worker':
        worker(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        main()
