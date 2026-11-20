"""FP32 vision computation for screenshots that overflow Jina's FP16 vision blocks."""
import time

import torch
from transformers import AutoModel

from jina_gpu import DEFAULT_MODEL, Encoder, dispatch_retrieval, keep_retrieval_adapters


class VisionFP32Encoder(Encoder):
    def __init__(self, model_path=DEFAULT_MODEL, mode="offload", max_pixels=602112, gpu_memory="4.8GiB"):
        if mode != "offload" or not torch.cuda.is_available():
            raise ValueError("FP32 vision requires CUDA offload mode")
        self.mode, self.dtype, self.device = mode, torch.float16, "cuda:0"
        self.vision_dtype = torch.float32
        torch.set_num_threads(8)
        started = time.perf_counter()
        self.model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True,
            local_files_only=True, torch_dtype=self.dtype, device_map="cpu", attn_implementation="sdpa")
        self.skip_unused_logits, self.quantized_layers = True, 0
        self.model.base_model.model.lm_head = torch.nn.Identity()
        keep_retrieval_adapters(self.model)
        visual = self.model.base_model.model.model.visual
        visual.float()
        original_forward = visual.forward

        def forward(*args, **kwargs):
            with torch.autocast("cuda", enabled=False):
                return original_forward(*args, **kwargs)

        visual.forward = forward
        self.model = dispatch_retrieval(self.model, gpu_memory)
        self.model.eval()
        self.model.processor.image_processor.max_pixels = max_pixels
        self.model.processor.image_processor.size["longest_edge"] = max_pixels
        self.load_seconds = time.perf_counter() - started
