from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from scripts.complete_paused_graph_index import CheckpointClient


def setup(tmp_path):
    image=tmp_path/'page.png';image.write_bytes(b'synthetic image bytes; mock encoder')
    client=SimpleNamespace(encode=Mock(return_value=np.ones((3,128),dtype=np.float32)),
        log=SimpleNamespace(emit=Mock()),endpoint='http://127.0.0.1:8796/v1/embeddings')
    return image,client,CheckpointClient(client,tmp_path/'cache')


def test_checkpoint_survives_reload_without_reencoding(tmp_path):
    image,client,cache=setup(tmp_path);identity={'fingerprint':'one','precision':'fp32'}
    first=cache.encode(image,'image',identity)
    reloaded=CheckpointClient(client,tmp_path/'cache')
    assert np.array_equal(first,reloaded.encode(image,'image',identity))
    assert client.encode.call_count==1
    assert reloaded.log.emit.call_args.kwargs['reused'] is True


def test_image_or_encoder_change_never_reuses_old_vector(tmp_path):
    image,client,cache=setup(tmp_path)
    cache.encode(image,'image',{'fingerprint':'one'})
    cache.encode(image,'image',{'fingerprint':'two'})
    image.write_bytes(b'changed image')
    cache.encode(image,'image',{'fingerprint':'two'})
    assert client.encode.call_count==3


def test_corrupt_checkpoint_fails_without_silent_reencoding(tmp_path):
    image,client,cache=setup(tmp_path);identity={'fingerprint':'one'}
    cache.encode(image,'image',identity)
    next((tmp_path/'cache').glob('*.npy')).write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='integrity'):
        cache.encode(image,'image',identity)
    assert client.encode.call_count==1
