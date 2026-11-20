import json
import numpy as np
import pytest
from PIL import Image
from web2kg.common.storage import Graph
from web2kg.stage1.visual_index import tile_boxes, score_views, build_visual_index, retrieve_visual, activate_visual_index
from test_visual_navigation import FakeJina


def test_boxes_cover_every_pixel_and_overlap():
    assert tile_boxes(20, 900) == []
    boxes = tile_boxes(20, 2001)
    assert boxes == [(0, 0, 20, 900), (0, 772, 20, 1672), (0, 1544, 20, 2001)]
    with pytest.raises(ValueError):
        tile_boxes(1, 2, 10, 10)


def test_crops_do_not_stitch_unrelated_query_tokens():
    q = np.eye(128, dtype=np.float32)[:2]
    vectors = {'p': q[:1], 'a': q[:1], 'b': q[1:]}
    score, view = score_views(q, vectors, 'p', [{'vector':'a'}, {'vector':'b'}])
    assert score == pytest.approx(.5)  # Concatenating crops would falsely yield 1.
    assert view['vector'] == 'a'


def test_round_trip_switch_reuses_whole_and_preserves_images(tmp_path):
    g, client = Graph(tmp_path), FakeJina()
    g.data.update(root='a', start_url='http://test/a')
    g.add_node({'id':'a','url':'http://test/a','title':'A','headings':[], 'actions':[], 'text':''}, [])
    path=tmp_path/'snapshots/a/page.png'; path.parent.mkdir(parents=True)
    Image.new('RGB',(20,2001),'white').save(path); original=path.read_bytes()
    whole=build_visual_index(g,client)
    client.calls.clear()
    tiled=build_visual_index(g,client,tiles=True)
    assert len(client.calls)==3
    assert len(tiled['tiling']['pages']['a'])==3
    assert path.read_bytes()==original
    hits=retrieve_visual(g,'test',10,client)
    assert len(hits)==1 and 'box' in hits[0]['matched_view']
    client.calls.clear()
    restored=build_visual_index(g,client,reuse_whole=True)
    assert not client.calls and 'tiling' not in restored
    assert (tmp_path/'index-tiled.json').exists()
    assert whole['images_sha256']==restored['images_sha256']
    activate_visual_index(g,client,'tiled')
    assert json.loads((tmp_path/'index.json').read_text())['tiling']
    activate_visual_index(g,client,'whole')
    assert not client.calls
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='screenshot changed'):
        activate_visual_index(g,client,'tiled')

def test_crop_preview_uses_original_image(tmp_path):
    from queue import Queue
    from threading import Thread
    from playwright.sync_api import sync_playwright, expect
    from web2kg.evaluation.manual import ManualSession
    from web2kg.visualization.manual import create_server
    g=Graph(tmp_path/'graph')
    g.data.update(root='a', start_url='http://test/a')
    g.add_node({'id':'a','url':'http://test/a','title':'Test','headings':[], 'actions':[], 'text':''}, [])
    path=g.directory/'snapshots/a/page.png'; path.parent.mkdir(parents=True)
    Image.new('RGB',(20,2001),'red').save(path); g.save()
    build_visual_index(g,FakeJina(),tiles=True)
    assert json.loads((g.directory/'index-whole.json').read_text())['score']=='mean_token_max_cosine'
    session=ManualSession(g,tmp_path/'review')
    session.data['queries'].append({'id':'q','text':'test','time':'2026-10-09T00:00:00Z',
        'top_k':1,'seconds':0,'success':True,'candidates':[{'id':'a','rank':1,'score':.5,'text':'',
        'matched_view':{'box':[0,772,20,1672]}}]})
    server=create_server(session,Queue(),0); thread=Thread(target=server.serve_forever,daemon=True); thread.start()
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(); page=browser.new_page(); errors=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            expect(page.locator('#cards')).to_contain_text('y=772–1672')
            page.get_by_role('button',name='查看命中切片',exact=True).click()
            expect(page.locator('#preview canvas')).to_be_visible()
            assert page.locator('#preview canvas').evaluate('(c)=>[c.width,c.height,...c.getContext("2d").getImageData(0,0,1,1).data]')==[20,900,255,0,0,255]
            assert not errors
            browser.close()
    finally:
        server.shutdown();server.server_close();thread.join()
