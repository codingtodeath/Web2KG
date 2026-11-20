from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from web2kg.common.interaction import VERSION, state_id
from web2kg.common.storage import Graph, EventLog
from web2kg.common.state_equivalence import StateResolver, small_difference, judgment_payload
from web2kg.common.replay import verify
from web2kg.stage1.jev_equivalence import JevEquivalenceClient


def page(token='secret-one', **changes):
    result = dict(identity_version=VERSION, url='https://example.test/', headings=['Home'],
        regions=[], text='Account menu', actions=[dict(id='exit',role='button',name='Log out',
        method='post',form_action='/logout?token=' + token, state={}, occurrence=0,match_count=1)], **changes)
    result['id'] = state_id(result)
    return result


def setup(tmp_path, decision='same'):
    graph = Graph(tmp_path)
    saved = page(); graph.nodes[saved['id']] = saved
    judge = SimpleNamespace(json=Mock(return_value={'decision': decision, 'score': .99}),
                            last_request_id='fixture', log=EventLog(tmp_path))
    return graph, saved, judge, StateResolver(graph, judge)


@pytest.mark.parametrize('form', [False, True])
def test_canonical_state_click_uses_current_dom_query_not_saved_query(tmp_path, form):
    from web2kg.common.browser import Browser
    from urllib.parse import urlsplit, parse_qs
    log = EventLog(tmp_path / 'browser')
    graph = Graph(tmp_path / 'graph')
    control = ('<form method="post" action="/confirm?token=old-value"><button>Continue</button></form>'
               if form else '<a href="/confirm?token=old-value">Continue</a>')
    submissions = []
    with Browser('about:blank', log, identity_mode='interaction') as browser:
        def route(request):
            if '/confirm?' in request.request.url:
                submissions.append(request.request.url)
                request.fulfill(body='<h1>Done</h1>', content_type='text/html')
            else:
                request.fulfill(body='<h1>Review</h1>' + control, content_type='text/html')
        browser.page.route('https://fixture.test/**', route)
        browser.page.goto('https://fixture.test/')
        before = browser.observe()
        action = next(a for a in before['actions'] if a['name'] == 'Continue')
        graph.nodes[before['id']] = deepcopy(before)
        judge = SimpleNamespace(json=Mock(return_value={'decision':'same','score':.99}),
                                last_request_id='fixture', log=log)
        resolver = StateResolver(graph, judge)
        browser.state_resolver = resolver
        browser.page.eval_on_selector('form' if form else 'a',
            "e => e.setAttribute(" + ("'action'" if form else "'href'") + ", '/confirm?token=new-value')")
        observed = browser.observe()
        normalized = resolver.resolve(observed, learn=True)
        assert normalized['id'] == before['id'] and normalized['raw_state_id'] != before['id']
        field = 'form_action' if form else 'href'
        current = next(a for a in normalized['actions'] if a['name'] == 'Continue')
        assert 'new-value' in current[field] and 'old-value' in action[field]
        # Replay passes the recorded action, but Browser.act must click the live DOM control.
        browser.act(action)
        assert len(submissions) == 1
        assert parse_qs(urlsplit(submissions[0]).query)['token'] == ['new-value']
        assert judge.json.call_count == 1


def test_one_judgment_reused_for_unseen_values_reload_and_discovery(tmp_path):
    graph, saved, judge, resolver = setup(tmp_path)
    original = deepcopy(saved)
    result = resolver.resolve(page('secret-two'), learn=True)
    assert result['id'] == saved['id'] and result['raw_state_id'] != saved['id']
    assert resolver.resolve(page('secret-three'), learn=True)['id'] == saved['id']
    assert judge.json.call_count == 1
    loaded = StateResolver(Graph.load(tmp_path))
    assert loaded.resolve(page('secret-four'))['id'] == saved['id']
    assert len(graph.nodes) == 1 and saved == original
    payload = judge.json.call_args.args[1]
    assert 'secret-one' not in json.dumps(payload) and 'secret-two' not in json.dumps(payload)
    assert payload['differences'][0]['parameter'] == 'token'


@pytest.mark.parametrize('field', ['url','headings','text','state','name','endpoint','extra_control','query_key','other_value'])
def test_rule_cannot_hide_other_changes(tmp_path, field):
    graph, saved, judge, resolver = setup(tmp_path)
    resolver.resolve(page('secret-two'), learn=True)
    changed = page('secret-three')
    if field == 'url': changed['url'] += 'different'
    elif field == 'headings': changed['headings'] = ['Subscribed']
    elif field == 'text': changed['text'] = 'Different account'
    elif field == 'state': changed['actions'][0]['state']['expanded'] = True
    elif field == 'name': changed['actions'][0]['name'] = 'Unsubscribe'
    elif field == 'endpoint': changed['actions'][0]['form_action'] = '/delete?token=secret-three'
    elif field == 'extra_control': changed['actions'].append(deepcopy(changed['actions'][0]))
    elif field == 'query_key': changed['actions'][0]['form_action'] = '/logout?object=secret-three'
    elif field == 'other_value': changed['actions'][0]['form_action'] += '&account=2'
    changed['id'] = state_id(changed)
    assert resolver.resolve(changed, learn=True)['id'] != saved['id']
    assert judge.json.call_count == 1


@pytest.mark.parametrize('decision', ['different','uncertain'])
def test_rejection_never_creates_rule_or_repeats_same_judgment(tmp_path, decision):
    _, saved, judge, resolver = setup(tmp_path, decision)
    changed = page('secret-two')
    assert resolver.resolve(changed, learn=True)['id'] != saved['id']
    assert resolver.resolve(changed, learn=True)['id'] != saved['id']
    assert judge.json.call_count == 1 and not resolver.data['rules']


def test_verification_uses_rules_even_when_saved_argument_omitted(tmp_path):
    _, saved, judge, resolver = setup(tmp_path)
    browser = SimpleNamespace(state_resolver=resolver, observe=lambda: page('secret-two'),
                              page=SimpleNamespace(wait_for_timeout=lambda _: None))
    assert verify(browser, saved['id'], saved)['id'] == saved['id']
    browser.observe = lambda: resolver.resolve(page('another-secret'))
    assert verify(browser, saved['id'])['id'] == saved['id']
    assert judge.json.call_count == 1


def test_multiple_slots_preserve_order_and_other_values():
    left, right = page(), page()
    left['actions'][0]['form_action'] = '/logout?token=A&object=1&nonce=B'
    right['actions'][0]['form_action'] = '/logout?token=C&object=1&nonce=D'
    assert len(small_difference(left, right)) == 2
    right['actions'][0]['form_action'] = '/logout?nonce=D&object=1&token=C'
    assert small_difference(left, right) is None
    right['actions'][0]['form_action'] = '/logout?token=C&object=2&nonce=D'
    assert small_difference(left, right) is None


def test_typed_judge_logs_model_and_rejects_malformed_without_retry(tmp_path):
    env = tmp_path / '.env'; env.write_text('api_key=private-key')
    client = JevEquivalenceClient(env, EventLog(tmp_path / 'calls'), 2, retry_connections=True,
                                  proxy='http://127.0.0.1:7890')
    payload = judgment_payload(page(), page('secret-two'), small_difference(page(), page('secret-two')))
    response = dict(model='jev-test', answers={'equivalent': {'type':'noul','noul':.99}},
                    usage={'input_tokens':10,'output_tokens':2})
    client.session.post = Mock(return_value=SimpleNamespace(status_code=200, json=lambda: response))
    assert client.json('',payload,{},'state_equivalence')['decision'] == 'same'
    response['answers']['equivalent']['noul'] = float('nan')
    from web2kg.common.responses import ResponsesError
    with pytest.raises(ResponsesError): client.json('',payload,{},'state_equivalence')
    assert client.session.post.call_count == 2
    for path in (tmp_path / 'calls').rglob('*.json'):
        assert 'secret-one' not in path.read_text() and 'private-key' not in path.read_text()


@pytest.mark.parametrize('score,decision',[(.79,'uncertain'),(.8,'same'),(.81,'same'),(.1,'different')])
def test_equivalence_threshold_boundary(tmp_path, score, decision):
    env = tmp_path / '.env'; env.write_text('api_key=private-key')
    client = JevEquivalenceClient(env, EventLog(tmp_path / 'calls'), 1)
    result = client.parse_result({'model':'jev-test','answers':{'equivalent':{'type':'noul','noul':score}}}, {}, tmp_path)
    assert client.identity['threshold'] == .8
    assert result['decision'] == decision


def test_actor_navigation_reuses_rules_at_source_and_target_without_model(tmp_path):
    from web2kg.stage2.navigation import RouteNavigator
    graph, source, judge, resolver = setup(tmp_path / 'graph')
    graph.data.update(root=source['id'],start_url=source['url'])
    target = page(); target['url'] += 'next'; target['id'] = state_id(target)
    graph.nodes[target['id']] = target
    resolver.resolve(page('second'), learn=True)
    after = deepcopy(target); after['actions'][0]['form_action']='/logout?token=third'; after['id']=state_id(after)
    resolver.resolve(after, learn=True)
    action={'kind':'click','role':'link','name':'Next','href':target['url']}
    graph.add_edge(source['id'],target['id'],action)
    graph.data['edges'][0].update(replayable=True,replay_policy='luna-v1')
    observed = [page('fourth')]
    browser=SimpleNamespace(start_url=source['url'],log=judge.log,
        observe=lambda *args: observed[0],act=lambda *args:observed.__setitem__(0,after))
    nav=RouteNavigator(browser,graph,EventLog(tmp_path/'navigation'))
    assert nav.identify(observed[0]) == source['id']
    assert nav.replay([graph.data['edges'][0]])['success']
    assert nav.identify(after) == target['id']
    assert judge.json.call_count == 2


def test_conflicting_existing_nodes_are_not_silently_merged(tmp_path):
    graph, source, judge, resolver=setup(tmp_path)
    second=page('second'); graph.nodes[second['id']]=second
    result=resolver.resolve(page('third'),learn=True,expected=source['id'])
    assert result['id'] not in graph.nodes and judge.json.call_count == 0


def test_switching_graph_on_same_browser_does_not_leak_learned_slots(tmp_path):
    from web2kg.common.replay import bind_resolver
    from web2kg.stage2.navigation import RouteNavigator
    graph, saved, judge, resolver = setup(tmp_path / 'first')
    resolver.resolve(page('second'), learn=True)
    browser = SimpleNamespace(state_resolver=resolver,start_url=saved['url'],log=judge.log)
    other = Graph(tmp_path / 'other')
    assert resolver.resolve(page('third'))['id'] == saved['id']
    bind_resolver(browser, other)
    assert browser.state_resolver is None
    browser.state_resolver = resolver
    RouteNavigator(browser,other,EventLog(tmp_path/'navigation'))
    assert browser.state_resolver is None


def test_full_context_keeps_late_controls_regions_text_and_exact_diff_addresses():
    before = page()
    before['regions'] = [['navigation','Account',{}],['main','Article',{}]]
    before['text'] = 'Article content explaining the account settings and menu.'
    before['actions'] = [dict(role='link',name=f'Article {i}',href=f'/article?id={i}',
                             region={'role':'main','name':'Article'}) for i in range(50)] + before['actions']
    before['actions'][-1]['region']={'role':'navigation','name':'Account'}
    after=deepcopy(before);after['actions'][-1]['form_action']='/logout?token=secret-two'
    payload=judgment_payload(before,after,small_difference(before,after))
    assert len(payload['before']['controls']) == len(payload['after']['controls']) == 51
    assert payload['before']['visible_text'] == before['text']
    assert payload['before']['regions'] == before['regions']
    assert payload['before']['controls'][-1]['region'] == before['actions'][-1]['region']
    diff=payload['differences'][0]
    assert diff['control_ref'] == payload['before']['controls'][-1]['ref'] == 'C0050'
    def at(value, pointer):
        for key in pointer.strip('/').split('/'):
            value=value[int(key)] if isinstance(value,list) else value[key]
        return value
    assert at(payload['before'],diff['context_pointer']) == diff['before_value']
    assert at(payload['after'],diff['context_pointer']) == diff['after_value']
    reconstructed=deepcopy(payload['before'])
    reconstructed['controls'][-1]['form_action']['query'][0]['value']=diff['after_value']
    assert reconstructed == payload['after']
    assert payload['coverage']['controls_omitted'] == 0
    assert 'secret-one' not in json.dumps(payload) and 'secret-two' not in json.dumps(payload)
