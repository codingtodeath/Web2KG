"""Local Jina V4 inference; keep downloaded model code and old indices unchanged."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel


DEFAULT_MODEL = Path("D:/Agent/MyCode/browser-use/webNavigator/models/jina-embeddings-v4")


def quantize_text(model, kind="nf4"):
    """Quantize base text linears, keeping task adapters, vision and projector FP16."""
    import bitsandbytes as bnb

    count = 0
    for name, module in list(model.named_modules()):
        if ".language_model.layers." not in name or not hasattr(module, "base_layer"):
            continue
        original = module.base_layer
        if not isinstance(original, torch.nn.Linear):
            continue
        if kind == "int8":
            layer = bnb.nn.Linear8bitLt(original.in_features, original.out_features,
                                       bias=original.bias is not None, has_fp16_weights=False, threshold=6.0)
            layer.weight = bnb.nn.Int8Params(original.weight.detach(), requires_grad=False, has_fp16_weights=False)
        else:
            layer = bnb.nn.Linear4bit(original.in_features, original.out_features,
                                      bias=original.bias is not None, compute_dtype=torch.float16,
                                      compress_statistics=True, quant_type="nf4")
            layer.weight = bnb.nn.Params4bit(original.weight.detach(), requires_grad=False,
                                           compress_statistics=True, quant_type="nf4")
        if original.bias is not None:
            layer.bias = original.bias
        module.base_layer = layer
        count += 1
    if not count:
        raise RuntimeError("No Jina text base linears found; refusing an unquantized GPU load")
    return count


def keep_retrieval_adapters(model):
    """Drop unused task weights; retrieval GEMMs already run under FP16 autocast."""
    for module in list(model.modules()):
        for attribute in ("lora_A", "lora_B"):
            for tasks in getattr(module, attribute, {}).values():
                for task in list(tasks.keys()):
                    if task != "retrieval":
                        del tasks[task]
                tasks["retrieval"].to(dtype=torch.float16)


def dispatch_retrieval(model, gpu_memory):
    """Prioritize text weights; stream individual vision blocks only for screenshots."""
    from accelerate import dispatch_model, infer_auto_device_map

    base = model.base_model.model
    language = infer_auto_device_map(base.model.language_model,
                                    max_memory={0: gpu_memory, "cpu": "12GiB"},
                                    no_split_module_classes=["Qwen2_5_VLDecoderLayer"])
    prefix = "base_model.model.model.language_model"
    mapping = {prefix + ("." + key if key else ""): value for key, value in language.items()}
    visual_prefix = "base_model.model.model.visual"
    for key, module in base.model.visual.named_children():
        if key == "blocks":
            mapping.update({f"{visual_prefix}.blocks.{i}": "cpu" for i in range(len(module))})
        else:
            mapping[f"{visual_prefix}.{key}"] = "cpu"
    mapping.update({"base_model.model.multi_vector_projector": 0, "base_model.model.lm_head": 0})
    return dispatch_model(model, device_map=mapping, main_device=0)


class Encoder:
    def __init__(self, model_path=DEFAULT_MODEL, mode="offload", max_pixels=602112, gpu_memory="4.8GiB"):
        self.mode = mode
        torch.set_num_threads(8)
        self.dtype = torch.bfloat16 if mode == "cpu" else torch.float16
        self.device = "cpu" if mode == "cpu" else "cuda:0"
        if mode != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use the dedicated Conda environment")
        started = time.perf_counter()
        self.model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True,
                                              local_files_only=True, torch_dtype=self.dtype,
                                              device_map="cpu", attn_implementation="sdpa")
        # Jina only consumes hidden states; vocabulary logits never enter either embedding.
        self.skip_unused_logits = mode != "cpu"
        if self.skip_unused_logits:
            self.model.base_model.model.lm_head = torch.nn.Identity()
        self.quantized_layers = 0
        if mode in ("nf4", "int8"):
            if mode == "int8":
                keep_retrieval_adapters(self.model)
            self.quantized_layers = quantize_text(self.model, mode)
            self.model.to(self.device)
        elif mode == "offload":
            keep_retrieval_adapters(self.model)
            self.model = dispatch_retrieval(self.model, gpu_memory)
        self.model.eval()
        self.model.processor.image_processor.max_pixels = max_pixels
        self.model.processor.image_processor.size["longest_edge"] = max_pixels
        self.load_seconds = time.perf_counter() - started

    def encode(self, values, kind="text", query=True):
        total_started = time.perf_counter()
        processor = self.model.processor
        if kind == "image":
            images = [Image.open(value).convert("RGB") for value in values]
            batch = processor.process_images(images)
        else:
            batch = processor.process_texts(values, prefix="Query" if query else "Passage")
        batch = {key: value.to(self.device) for key, value in batch.items()}
        if self.mode != "cpu":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        autocast = torch.autocast(self.device.split(":")[0], dtype=self.dtype)
        with torch.inference_mode(), autocast:
            output = self.model(**batch, task_label="retrieval")
        mask = batch["attention_mask"].bool()
        vectors = [row[valid].float().cpu().numpy() for row, valid in zip(output.multi_vec_emb, mask)]
        if self.mode != "cpu":
            torch.cuda.synchronize()
        stats = {"seconds": time.perf_counter() - started, "total_seconds": time.perf_counter() - total_started,
                 "tokens": [len(v) for v in vectors],
                 "peak_gpu_bytes": torch.cuda.max_memory_allocated() if self.mode != "cpu" else 0,
                 "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if self.mode != "cpu" else 0}
        if not all(np.isfinite(v).all() for v in vectors):
            raise ValueError("Non-finite embedding; configuration rejected")
        return vectors, stats


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=["cpu", "offload", "nf4", "int8"], default="offload")
    parser.add_argument("--max-pixels", type=int, default=602112)
    parser.add_argument("--gpu-memory", default="4.8GiB")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", default=[])
    return parser.parse_args()


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    queries = ["Find the forum directory", "Find account profile settings", "Read a discussion and its comments"]
    log = {"mode": args.mode, "max_pixels": args.max_pixels, "torch": torch.__version__,
           "cuda": torch.version.cuda, "queries": queries, "images": [str(p) for p in args.image],
           "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None, "runs": []}
    try:
        encoder = Encoder(args.model_path, args.mode, args.max_pixels, args.gpu_memory)
        log.update(load_seconds=encoder.load_seconds, quantized_layers=encoder.quantized_layers,
                   skip_unused_logits=encoder.skip_unused_logits)
        for repeat in range(2):
            for index, query in enumerate(queries):
                vectors, stats = encoder.encode([query])
                np.save(args.output / f"query-{index}-{repeat}.npy", vectors[0])
                log["runs"].append(dict(kind="text", index=index, repeat=repeat, **stats))
                (args.output / "metrics.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
                print(f"text {index} repeat {repeat}: {stats}", flush=True)
            for index, path in enumerate(args.image):
                if args.mode == "cpu" and repeat == 1:
                    continue
                vectors, stats = encoder.encode([str(path)], kind="image")
                np.save(args.output / f"image-{index}-{repeat}.npy", vectors[0])
                log["runs"].append(dict(kind="image", index=index, repeat=repeat, **stats))
                (args.output / "metrics.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
                print(f"image {index} repeat {repeat}: {stats}", flush=True)
        log["status"] = "passed"
    except Exception as error:
        log.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (args.output / "metrics.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
