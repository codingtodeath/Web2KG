import copy
import importlib
import json
from types import SimpleNamespace

from web2kg.stage2.agent import load_runtime


def processor():
    load_runtime()
    return importlib.import_module('browser_env.processors').TextObervationProcessor


def node(key, role, name='', children=(), **kwargs):
    kwargs.setdefault('properties', [])
    return dict(nodeId=str(key), role={'value': role}, name={'value': name},
                childIds=list(map(str, children)), backendDOMNodeId=key,
                union_bound=None, **kwargs)


def test_ax_text_preserves_hidden_parent_depth_properties_and_order():
    nodes = [node(1, 'RootWebArea', 'Fixture', [2, 6, 999]),
             node(2, 'generic', children=[3, 4, 5]),
             node(3, 'link', '中文', url='https://fixture.test/comments'),
             node(4, 'button', 'Save', properties=[{'name': 'disabled', 'value': {'value': True}}]),
             node(5, 'generic'), node(6, 'StaticText', 'Done')]
    text, info, root = processor().parse_accessibility_tree(nodes)
    assert text == ("[1] RootWebArea 'Fixture'\n"
                    "\t[3] link '中文' url: https://fixture.test/comments\n"
                    "\t[4] button 'Save' disabled: True\n\t[6] StaticText 'Done'")
    assert list(info) == ['1', '3', '4', '6']
    assert [n.node_id for n in root.children] == [3, 4, 6]
    assert all(n.depth == 1 and n.parent is root for n in root.children)
    assert root.children[1].properties == {'disabled': True}


def test_ax_wide_page_keeps_all_nodes_and_refreshes():
    nodes = [node(1, 'RootWebArea', 'Wide', range(2, 2002))]
    nodes.extend(node(i, 'link', f'Comment {i}', url=f'https://fixture.test/{i}')
                 for i in range(2, 2002))
    parse = processor().parse_accessibility_tree
    text, info, root = parse(copy.deepcopy(nodes))
    assert len(text.splitlines()) == len(info) == 2001
    assert [n.node_id for n in root.children] == list(range(2, 2002))
    nodes[-1]['url'] = 'https://fixture.test/changed'
    refreshed, _, updated = parse(nodes)
    assert refreshed != text
    assert updated.children[-1].properties['url'].endswith('/changed')


def test_unfinished_trajectory_preserves_full_data_and_each_checkpoint(tmp_path):
    save = load_runtime().webNavigator.save_unfinished_trajectory
    run = SimpleNamespace(log_dir=str(tmp_path), task_id='fixture', trajectory=[])
    path = tmp_path / 'fixture_unfinished.json'
    for step in range(3):
        run.trajectory.append({'action': f'click [{step}]', 'text': '中文\n"quoted"',
                               'values': [None, True, 1.25]})
        save(run)
        assert json.loads(path.read_text()) == dict(task_id='fixture', status='unfinished',
                                                    num_steps=step + 1, trajectory=run.trajectory)
    previous = path.read_bytes()
    run.trajectory.append({'unsupported': object()})
    save(run)
    assert path.read_bytes() == previous
