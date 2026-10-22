"""Load a merged Dohnuts DecisionModel on CUDA or CPU without editing vendor/dohnuts.

On CUDA this is exactly the vendor path. The vendor adapter hardcodes CUDA and
the FLA/Triton fused kernels, so on CPU the same model is assembled here with
the same BF16 weights, linear patch embedding, LoRA, head, and merge, but with
transformers' own PyTorch Qwen3.5 RMSNorm/SwiGLU/gated-delta ops instead of the
fused kernels (same math, different kernels).
"""

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel

from dohnuts.adapters import Qwen35Adapter
from dohnuts.kernels import enable_linear_patch_embedding
from dohnuts.model import DecisionModel
from dohnuts.train import load_checkpoint

QWEN = ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")
DOHNUTS = ("PsiACE/Dohnuts-0.1.0-0.8B", "959828a3d9cf3dacbd1491e11bb1cd36d12709ac")


def snapshot(repo_revision):
    from huggingface_hub import snapshot_download

    repo, revision = repo_revision
    return Path(snapshot_download(repo, revision=revision))


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _cpu_decision_model(base_path):
    model = DecisionModel.__new__(DecisionModel)
    nn.Module.__init__(model)
    adapter = Qwen35Adapter()
    model.adapter = adapter
    model.base_path = Path(base_path)
    model.processor = adapter.processor(base_path)
    backbone = AutoModel.from_pretrained(
        base_path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    )
    backbone.requires_grad_(False)
    backbone.config.use_cache = False
    enable_linear_patch_embedding(backbone.visual)
    backbone._dohnuts_image_cache = OrderedDict()
    model.backbone = backbone
    model.head = nn.Linear(adapter.hidden_size(backbone), 1, bias=False, dtype=torch.float32)
    model.marker_id = model.processor.tokenizer.convert_tokens_to_ids(adapter.marker)
    return model


def load_decision_model(base_path, *, checkpoint=None, adapter_dir=None, device=None):
    """checkpoint: a trainer best.pt (ActOne). adapter_dir: an exported Dohnuts adapter."""
    device = device or default_device()
    model = DecisionModel(base_path) if device == "cuda" else _cpu_decision_model(base_path)
    model.enable_lora(checkpointing=False)
    if checkpoint is not None:
        load_checkpoint(model, checkpoint)
    elif adapter_dir is not None:
        model.load_adapter(adapter_dir)
    else:
        raise ValueError("pass checkpoint= or adapter_dir=")
    model.merge()
    model.device = device
    return model
