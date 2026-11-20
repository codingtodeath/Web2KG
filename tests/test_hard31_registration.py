from scripts import register_hard31_225 as reg


def test_fixed_previous225_lineage_and_exact_remaining31():
    ref=reg.run.read(reg.REFERENCE / 'protocol.json')
    old,new=reg.selected_inputs(ref)
    assert [r['task_id'] for r in old]==[646,598,625,596,721,31,730,409,651,644]
    ids=[r['task_id'] for r in new]
    assert len(ids)==len(set(ids))==31 and ids[0]==735
    assert 724 not in ids and not set(ids)&{r['task_id'] for r in old}
    assert all('225' in r['directory'] or 'background' in r['directory'] or 'parser' in r['directory'] for r in old)
    assert old[6]['state']['status']=='needs_attention'  # no rewriting the old730 failure
    assert reg.SEED.name=='execution-graph' and reg.SEED.parent.name=='644'
