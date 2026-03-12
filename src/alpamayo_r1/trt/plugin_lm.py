# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TRT-LLM plugin-based LM compilation and inference utilities.

This module provides FP16 TRT compilation for the Alpamayo pipeline using
TRT-LLM attention plugins for the language model. The pipeline is:

    Vision (TRT) → LM with plugin attention (TRT) → Diffusion/Expert (TRT)

All components run in FP16 to match the plugin's precision constraint.

Usage:
    from alpamayo_r1.trt.plugin_lm import (
        compile_all_engines, run_trt_inference, run_pytorch_inference,
    )
"""

from __future__ import annotations

import copy
import gc
import json
import logging
import os
import time
from typing import Any, List, Tuple

import einops
import numpy as np
import torch
import torch.nn as nn
import torch_tensorrt

from alpamayo_r1 import helper
from alpamayo_r1.models.token_utils import (
    replace_padding_after_eos,
    to_special_token,
)
from alpamayo_r1.trt.diffusion import (
    _build_diffusion_module,
    _build_trt_input_specs,
    _export_diffusion_module,
    _make_sample_inputs,
)
from alpamayo_r1.trt.vision import (
    VisualFixedGrid,
    _RepeatCollapseVisionWrapper,
    _export_vision_module,
    _patch_qwen3vl_vision_attention,
)

logger = logging.getLogger(__name__)

FP16 = torch.float16
DEVICE = torch.device("cuda:0")


# ============================================================
# Utility functions
# ============================================================

def compute_min_ade(pred_xyz: torch.Tensor, data: dict) -> float:
    """Compute minimum Average Displacement Error (meters)."""
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].numpy()
    pred_xy = pred_xyz.detach().cpu().numpy()[0, 0, :, :, :2]
    d = np.linalg.norm(pred_xy - gt_xy[None, :, :], axis=-1)
    return float(d.mean(axis=-1).min())


def sample_token(
    logits: torch.Tensor,
    traj_token_offset: int,
    traj_vocab_size: int,
    temperature: float = 0.6,
    top_p: float = 0.98,
) -> torch.Tensor:
    """Temperature-scaled top-p (nucleus) sampling with trajectory token masking."""
    logits = logits[:, -1, :].float()
    logits[:, traj_token_offset : traj_token_offset + traj_vocab_size] = float("-inf")
    logits = logits / temperature
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove_mask = cum_probs - sorted_logits.softmax(dim=-1) >= top_p
    sorted_logits[remove_mask] = float("-inf")
    logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
    return torch.multinomial(logits.softmax(dim=-1), num_samples=1)


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """Compute max diff, mean diff, and cosine similarity between two tensors."""
    a_f, b_f = a.float().flatten(), b.float().flatten()
    diff = torch.abs(a_f - b_f)
    cos_sim = torch.nn.functional.cosine_similarity(
        a_f.unsqueeze(0), b_f.unsqueeze(0)
    ).item()
    return {
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
        "cosine_sim": cos_sim,
    }


def prepare_model_inputs(model, data: dict, device: str = "cuda") -> dict:
    """Prepare tokenized model inputs from raw data."""
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    return helper.to_device({
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"].clone(),
        "ego_history_rot": data["ego_history_rot"].clone(),
    }, device)


# ============================================================
# VLM preprocessing
# ============================================================

def run_vlm_preprocessing(model, model_inputs: dict, trt_vision=None):
    """
    Embed tokens, get image features, fuse, compute position_ids & rope_deltas.

    Returns:
        (input_ids, inputs_embeds, deepstack_image_embeds,
         visual_pos_masks, position_ids, rope_deltas)
    """
    tokenized_data = copy.deepcopy(model_inputs["tokenized_data"])
    input_ids = tokenized_data.pop("input_ids")
    input_ids = model.fuse_traj_tokens(input_ids, {
        "ego_history_xyz": model_inputs["ego_history_xyz"],
        "ego_history_rot": model_inputs["ego_history_rot"],
    })

    vlm_model = model.vlm.model
    lm_ref = vlm_model.language_model

    original_fwd = None
    if trt_vision is not None:
        original_fwd = vlm_model.visual.forward
        vlm_model.visual.forward = trt_vision.forward

    with torch.no_grad(), torch.autocast("cuda", dtype=FP16):
        inputs_embeds = lm_ref.embed_tokens(input_ids)
        pv = tokenized_data["pixel_values"].to(DEVICE)
        igt = tokenized_data["image_grid_thw"].to(DEVICE)
        image_embeds, ds_embeds = vlm_model.get_image_features(pv, igt)
        image_cat = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = vlm_model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_cat
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_cat)
        vis_masks = image_mask[..., 0]
        attn = tokenized_data.get("attention_mask")
        if attn is not None:
            attn = attn.to(DEVICE)
        position_ids, rope_deltas = vlm_model.get_rope_index(
            input_ids, igt, video_grid_thw=None, attention_mask=attn
        )

    if original_fwd is not None:
        vlm_model.visual.forward = original_fwd

    del pv, igt, image_embeds, image_cat, image_mask
    torch.cuda.empty_cache()
    return input_ids, inputs_embeds, ds_embeds, vis_masks, position_ids, rope_deltas


# ============================================================
# Plugin LM wrapper
# ============================================================

class PluginWrapperDSInput(nn.Module):
    """Plugin LM wrapper with deepstack as input tensor (reusable across clips)."""

    def __init__(self, lm: nn.Module, lm_head: nn.Module, num_ds: int):
        super().__init__()
        self.lm = lm
        self.lm_head = lm_head
        self.num_ds = num_ds

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        kv_caches: List[torch.Tensor],
        ctx_len: torch.Tensor,
        ds_stack: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        hidden = inputs_embeds
        seq_len = inputs_embeds.shape[1]
        new_kvs = []
        for i, layer in enumerate(self.lm.layers):
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            hidden, kv = layer.self_attn(
                hidden_states=hidden, past_key_value=kv_caches[i], ctx_len=ctx_len
            )
            hidden = residual + hidden
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden
            new_kvs.append(kv)
            if i < self.num_ds:
                hidden = hidden + ds_stack[i, :, :seq_len, :]
        hidden = self.lm.norm(hidden)
        return self.lm_head(hidden), new_kvs


class _FP16CastWrapper(nn.Module):
    """Casts inputs to FP16 before calling the TRT vision model."""

    def __init__(self, trt_model):
        super().__init__()
        self.trt_model = trt_model

    def forward(self, hidden_states, grid_thw=None):
        return self.trt_model(hidden_states.to(FP16), grid_thw)


# ============================================================
# TRT compilation
# ============================================================

def compile_vision_fp16(model, model_inputs: dict) -> nn.Module:
    """Compile TRT vision encoder in FP16."""
    logger.info("Compiling Vision TRT (FP16)...")
    _patch_qwen3vl_vision_attention()

    vis = copy.deepcopy(model.vlm.model.visual).to(dtype=FP16, device=DEVICE).eval()
    vis.config.attn_implementation = "sdpa"
    vis.config._attn_implementation = "sdpa"
    vis.config.use_cache = False

    pv = model_inputs["tokenized_data"]["pixel_values"].to(dtype=FP16, device=DEVICE)
    igt = model_inputs["tokenized_data"]["image_grid_thw"].to(device=DEVICE)

    wrapped = VisualFixedGrid(vis, igt).to(DEVICE).eval()
    ep = _export_vision_module(wrapped, (pv, None))
    trt_raw = torch_tensorrt.dynamo.compile(
        ep, (pv, None),
        truncate_double=True, min_block_size=1, use_python_runtime=True,
        immutable_weights=True, use_explicit_typing=True, use_fp32_acc=True,
    )
    trt_repeat = _RepeatCollapseVisionWrapper(
        trt_model=trt_raw,
        base_pixel_rows=int(pv.shape[0]),
        base_grid_rows=int(igt.shape[0]),
    ).eval()
    trt_vision = _FP16CastWrapper(trt_repeat).eval()

    with torch.no_grad():
        ref = wrapped(pv, None)
        trt = trt_vision(pv, None)
    d = tensor_diff(ref[0], trt[0])
    logger.info(f"  Vision: cos_sim={d['cosine_sim']:.6f}, max|Δ|={d['max_diff']:.6f}")
    logger.info("  ✅ Vision TRT compiled")

    del vis, wrapped
    torch.cuda.empty_cache(); gc.collect()
    return trt_vision


def compile_lm_plugin_fp16(
    model, S_input: int, position_ids, rope_deltas,
    num_ds_layers: int, max_seq_len: int, batch_size: int = 1,
):
    """
    Compile TRT plugin LM in FP16. Returns (trt_lm, embed_tokens_fp16, lm_config).
    """
    from plugin_utils import (
        PluginAttention, create_kv_caches, load_plugin,
        register_plugin_op, set_plugin_config_from_model,
    )
    logger.info(f"Compiling LM Plugin TRT (FP16, batch={batch_size})...")
    load_plugin(); register_plugin_op()

    lm_ref = model.vlm.model.language_model
    config = lm_ref.config
    hd, nkv, nl, hs = config.head_dim, config.num_key_value_heads, config.num_hidden_layers, config.hidden_size
    set_plugin_config_from_model(config, max_seq_len)

    lm = copy.deepcopy(lm_ref).to(dtype=FP16, device=DEVICE).eval()

    with torch.no_grad():
        d_eff = torch.arange(S_input, max_seq_len, device=DEVICE).float()
        d_eff += rope_deltas.to(DEVICE).float().squeeze()
        d_3d = d_eff.view(1, 1, -1).expand(3, 1, -1).long()
        full_pos = torch.cat([position_ids.to(DEVICE), d_3d], dim=2)
        cos, sin = lm.rotary_emb(torch.ones(1, device=DEVICE, dtype=FP16), full_pos)
        h2 = hd // 2
        rope_cache = torch.cat([cos[:, :max_seq_len, :h2].float(),
                                sin[:, :max_seq_len, :h2].float()], dim=-1)

    embed_tokens = copy.deepcopy(lm.embed_tokens)
    for i, layer in enumerate(lm.layers):
        layer.self_attn = PluginAttention(layer.self_attn, config, i, rope_cache)

    head = copy.deepcopy(model.vlm.lm_head).to(dtype=FP16, device=DEVICE).eval()
    wrapper = PluginWrapperDSInput(lm, head, num_ds_layers).to(DEVICE).eval()

    B = batch_size
    dummy_e = torch.randn(B, 3, hs, dtype=FP16, device=DEVICE)
    dummy_ctx = torch.tensor([3] * B, dtype=torch.int32, device=DEVICE)
    dummy_kvs = [torch.zeros(B, 2, nkv, max_seq_len, hd, dtype=FP16, device=DEVICE) for _ in range(nl)]
    dummy_ds = torch.zeros(num_ds_layers, B, max_seq_len, hs, dtype=FP16, device=DEVICE)

    seq_dim = torch.export.Dim("seq_len", min=1, max=max_seq_len)
    try:
        ep = torch.export.export(
            wrapper, args=(dummy_e, dummy_kvs, dummy_ctx, dummy_ds),
            dynamic_shapes={"inputs_embeds": {1: seq_dim}, "kv_caches": [{}]*nl, "ctx_len": {}, "ds_stack": {}},
            strict=False,
        )
    except Exception:
        ep = torch.export._trace._export(
            wrapper, (dummy_e, dummy_kvs, dummy_ctx, dummy_ds),
            dynamic_shapes={"inputs_embeds": {1: seq_dim}, "kv_caches": [{}]*nl, "ctx_len": {}, "ds_stack": {}},
            strict=False, prefer_deferred_runtime_asserts_over_guards=True,
        )
    trt_lm = torch_tensorrt.dynamo.compile(
        ep, inputs=[dummy_e, dummy_kvs, dummy_ctx, dummy_ds],
        enabled_precisions={torch.float32}, use_explicit_typing=True,
        use_fp32_acc=True, device=DEVICE, disable_tf32=True, min_block_size=1, dryrun=False,
    )

    del wrapper, dummy_e, dummy_kvs, dummy_ds, ep
    torch.cuda.empty_cache(); gc.collect()
    logger.info(f"  ✅ LM Plugin TRT compiled (batch={batch_size})")
    return trt_lm, embed_tokens, config


def compile_diffusion_fp16(
    model, min_prefix_len: int, max_prefix_len: int, batch_size: int = 1,
) -> nn.Module:
    """Compile TRT diffusion step in FP16."""
    logger.info(f"Compiling Diffusion TRT (FP16, batch={batch_size})...")

    module, cfg = _build_diffusion_module(model, dtype=FP16, device="cuda")
    sample_inputs, dyn_shapes, make_fn = _make_sample_inputs(
        cfg, min_prefix_len, max_prefix_len, FP16, "cuda", batch_size=batch_size,
    )

    with torch.no_grad():
        ref = module(*sample_inputs)
    exported = _export_diffusion_module(module, sample_inputs, dyn_shapes)
    opt = (min_prefix_len + max_prefix_len) // 2
    specs = _build_trt_input_specs(make_fn, min_prefix_len, opt, max_prefix_len)

    trt_diff = torch_tensorrt.dynamo.compile(
        exported, inputs=specs,
        use_explicit_typing=True, use_fp32_acc=True, truncate_double=True,
        min_block_size=1, use_python_runtime=True,
    )

    with torch.no_grad():
        trt_out = trt_diff(*sample_inputs)
    d = tensor_diff(ref, trt_out)
    logger.info(f"  Diffusion: cos_sim={d['cosine_sim']:.6f}, max|Δ|={d['max_diff']:.6f}")
    logger.info(f"  ✅ Diffusion TRT compiled (batch={batch_size})")

    del module, exported, sample_inputs
    torch.cuda.empty_cache(); gc.collect()
    return trt_diff


def compile_all_engines(model, model_inputs: dict, max_gen: int = 256, batch_size: int = 1):
    """
    Compile all TRT engines (vision, LM, diffusion).

    Returns:
        dict with keys: trt_vision, trt_lm, trt_diffusion, embed_tokens, lm_config,
                        S_input, max_seq_len, num_ds_layers, rope_deltas_ref
    """
    logger.info("=" * 60)
    logger.info("Compiling all TRT engines")
    logger.info("=" * 60)

    _, embeds, ds, _, pos_ids, rope_deltas = run_vlm_preprocessing(model, model_inputs)
    S_input = embeds.shape[1]
    num_ds = len(ds)
    max_seq_len = max(4096, S_input + max_gen + 100)

    trt_vision = compile_vision_fp16(model, model_inputs)
    trt_lm, embed_tokens, lm_config = compile_lm_plugin_fp16(
        model, S_input, pos_ids, rope_deltas, num_ds, max_seq_len, batch_size=batch_size,
    )
    trt_diffusion = compile_diffusion_fp16(
        model, S_input, S_input + max_gen + 10, batch_size=batch_size,
    )

    del embeds, ds
    torch.cuda.empty_cache(); gc.collect()
    logger.info("✅ All engines compiled")

    return {
        "trt_vision": trt_vision, "trt_lm": trt_lm, "trt_diffusion": trt_diffusion,
        "embed_tokens": embed_tokens, "lm_config": lm_config,
        "S_input": S_input, "max_seq_len": max_seq_len,
        "num_ds_layers": num_ds, "rope_deltas_ref": rope_deltas.clone(),
    }


# ============================================================
# Engine cache save / load  (`.engine` format)
# ============================================================
#
# Cache directory layout:
#     trt_vision.engine      raw TRT engine binary
#     trt_lm.engine          raw TRT engine binary
#     trt_diffusion.engine   raw TRT engine binary
#     engines.json           binding names + vision wrapper params
#     metadata.pt            embed_tokens, lm_config, scalars
#

def _find_trt_submodule(module: nn.Module):
    """Return the first TRT runtime submodule found in *module*'s tree.

    Handles both the Python runtime (``PythonTorchTensorRTModule``) and
    the C++ runtime (``TorchTensorRTModule``).
    """
    for _, submod in module.named_modules():
        if (
            hasattr(submod, "serialized_engine")
            and isinstance(getattr(submod, "serialized_engine", None), bytes)
        ):
            return submod
    return None


def _get_trt_binding_names(
    trt_sub: nn.Module,
) -> tuple[list[str], list[str]]:
    """Return ``(input_names, output_names)`` from a TRT submodule.

    The Python runtime stores them as ``input_names`` / ``output_names``,
    while the C++ runtime uses ``input_binding_names`` / ``output_binding_names``.
    """
    in_names = getattr(trt_sub, "input_names", None) or getattr(
        trt_sub, "input_binding_names", []
    )
    out_names = getattr(trt_sub, "output_names", None) or getattr(
        trt_sub, "output_binding_names", []
    )
    return list(in_names), list(out_names)


class _CachedVisionTRTModule(nn.Module):
    """Adaptor that accepts ``(hidden_states, grid_thw=None)`` and
    forwards only ``hidden_states`` to the underlying TRT module.

    The raw TRT engine returns a **flat** tuple ``(main_embed, ds_0, ds_1, …)``.
    The original compiled GraphModule preserves the pytree structure
    ``(main_embed, [ds_0, ds_1, …])``.  This adaptor repacks the flat output
    back into the expected ``(Tensor, list[Tensor])`` shape so that
    ``get_image_features`` can unpack it correctly.
    """

    def __init__(self, trt_module: nn.Module):
        super().__init__()
        self.trt_module = trt_module

    def forward(self, hidden_states: torch.Tensor, grid_thw=None):
        outputs = self.trt_module(hidden_states)
        if isinstance(outputs, (tuple, list)) and len(outputs) > 1:
            # Repack: (main_embed, ds_0, ds_1, …) → (main_embed, [ds_0, ds_1, …])
            return outputs[0], list(outputs[1:])
        return outputs


class _CachedLMTRTModule(nn.Module):
    """Adaptor that flattens / unflattens the KV-cache list so the
    underlying ``PythonTorchTensorRTModule`` sees flat positional args
    in the correct binding-name order."""

    def __init__(self, trt_module: nn.Module, num_layers: int,
                 input_names: list[str]):
        super().__init__()
        self.trt_module = trt_module
        self.num_layers = num_layers
        self._input_names = input_names

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        kv_caches: List[torch.Tensor],
        ctx_len: torch.Tensor,
        ds_stack: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        name2tensor: dict[str, torch.Tensor] = {
            "inputs_embeds": inputs_embeds,
            "ctx_len": ctx_len,
            "ds_stack": ds_stack,
        }
        for i in range(self.num_layers):
            name2tensor[f"kv_caches_{i}"] = kv_caches[i]
        flat = [name2tensor[n] for n in self._input_names]
        outputs = self.trt_module(*flat)
        logits = outputs[0]
        new_kvs = list(outputs[1 : 1 + self.num_layers])
        return logits, new_kvs


class _CachedDiffusionTRTModule(nn.Module):
    """Adaptor that reorders positional args to match the TRT engine's
    binding-name order (which may differ from the Python call order)."""

    def __init__(self, trt_module: nn.Module, input_names: list[str]):
        super().__init__()
        self.trt_module = trt_module
        self._input_names = input_names

    def forward(
        self, x: torch.Tensor, t: torch.Tensor,
        prefix_k: torch.Tensor, prefix_v: torch.Tensor,
        position_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        name2tensor = {
            "x": x, "t": t,
            "prefix_k": prefix_k, "prefix_v": prefix_v,
            "position_ids": position_ids, "attention_mask": attention_mask,
        }
        flat = [name2tensor[n] for n in self._input_names]
        return self.trt_module(*flat)


def save_engines_cache(engines: dict, cache_dir: str) -> None:
    """Save compiled TRT engines as ``.engine`` files + metadata.

    Directory layout::

        cache_dir/
            trt_vision.engine      # raw TRT engine binary
            trt_lm.engine          # raw TRT engine binary
            trt_diffusion.engine   # raw TRT engine binary
            engines.json           # binding names, vision wrapper params
            metadata.pt            # embed_tokens, lm_config, scalars
    """
    os.makedirs(cache_dir, exist_ok=True)

    engine_meta: dict[str, Any] = {}

    for name in ("trt_vision", "trt_lm", "trt_diffusion"):
        trt_sub = _find_trt_submodule(engines[name])
        if trt_sub is None:
            raise RuntimeError(
                f"No PythonTorchTensorRTModule found in engines['{name}']. "
                "Cannot extract serialized engine bytes."
            )

        engine_path = os.path.join(cache_dir, f"{name}.engine")
        engine_bytes = trt_sub.serialized_engine
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
        logger.info(
            f"  Saved {name} → {engine_path} "
            f"({len(engine_bytes) / 1024 / 1024:.1f} MB)"
        )

        in_names, out_names = _get_trt_binding_names(trt_sub)
        engine_meta[name] = {
            "input_names": in_names,
            "output_names": out_names,
            "engine_size": len(engine_bytes),
        }

    # Vision wrapper params
    engine_meta["vision_wrapper"] = {
        "base_pixel_rows": engines["trt_vision"].trt_model.base_pixel_rows,
        "base_grid_rows": engines["trt_vision"].trt_model.base_grid_rows,
    }

    json_path = os.path.join(cache_dir, "engines.json")
    with open(json_path, "w") as f:
        json.dump(engine_meta, f, indent=2)

    # PyTorch metadata (no TRT objects → pickling is safe)
    metadata = {
        "embed_tokens": engines["embed_tokens"],
        "lm_config": engines["lm_config"],
        "S_input": engines["S_input"],
        "max_seq_len": engines["max_seq_len"],
        "num_ds_layers": engines["num_ds_layers"],
        "rope_deltas_ref": engines["rope_deltas_ref"].cpu(),
    }
    meta_path = os.path.join(cache_dir, "metadata.pt")
    torch.save(metadata, meta_path)

    logger.info(f"✅ Engine cache saved to {cache_dir}")


def load_engines_cache(cache_dir: str) -> dict | None:
    """Load cached ``.engine`` files and reconstruct the engines dict.

    Returns the same dict structure as ``compile_all_engines()``,
    or ``None`` if the cache is missing / incomplete / corrupt.
    """
    required = [
        "trt_vision.engine", "trt_lm.engine", "trt_diffusion.engine",
        "engines.json", "metadata.pt",
    ]
    if not all(os.path.exists(os.path.join(cache_dir, f)) for f in required):
        logger.info(f"Engine cache incomplete at {cache_dir}")
        return None

    try:
        from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule

        # The LM engine contains the TRT-LLM AttentionPlugin.
        # The plugin must be registered BEFORE deserializing the engine.
        from plugin_utils import load_plugin, register_plugin_op, set_plugin_config_from_model
        load_plugin()
        register_plugin_op()

        with open(os.path.join(cache_dir, "engines.json")) as f:
            engine_meta = json.load(f)

        metadata = torch.load(
            os.path.join(cache_dir, "metadata.pt"),
            map_location=DEVICE, weights_only=False,
        )

        # Configure plugin with LM config (needed for AttentionPlugin)
        set_plugin_config_from_model(metadata["lm_config"], metadata["max_seq_len"])

        engines: dict = dict(metadata)

        for name in ("trt_vision", "trt_lm", "trt_diffusion"):
            engine_path = os.path.join(cache_dir, f"{name}.engine")
            with open(engine_path, "rb") as f:
                engine_bytes = f.read()

            binding = engine_meta[name]

            # Validate file integrity
            expected_size = binding.get("engine_size")
            if expected_size is not None and len(engine_bytes) != expected_size:
                raise RuntimeError(
                    f"{name}.engine is corrupted: expected {expected_size} bytes, "
                    f"got {len(engine_bytes)}. Delete the cache and recompile."
                )
            trt_sub = PythonTorchTensorRTModule(
                serialized_engine=engine_bytes,
                input_binding_names=binding["input_names"],
                output_binding_names=binding["output_names"],
            )

            in_names = binding["input_names"]

            if name == "trt_vision":
                wp = engine_meta["vision_wrapper"]
                trt_repeat = _RepeatCollapseVisionWrapper(
                    trt_model=_CachedVisionTRTModule(trt_sub),
                    base_pixel_rows=wp["base_pixel_rows"],
                    base_grid_rows=wp["base_grid_rows"],
                ).eval()
                engines[name] = _FP16CastWrapper(trt_repeat).eval()
            elif name == "trt_lm":
                num_layers = metadata["lm_config"].num_hidden_layers
                engines[name] = _CachedLMTRTModule(
                    trt_sub, num_layers, in_names,
                )
            else:  # diffusion
                engines[name] = _CachedDiffusionTRTModule(
                    trt_sub, in_names,
                )

            sz_mb = len(engine_bytes) / 1024 / 1024
            logger.info(f"  Loaded {name} ← {engine_path} ({sz_mb:.1f} MB)")

        logger.info(f"✅ Engine cache loaded from {cache_dir}")
        return engines
    except Exception as e:
        logger.warning(f"Failed to load engine cache from {cache_dir}: {e}")
        return None


# ============================================================
# Inference — TRT plugin (batch decode)
# ============================================================

@torch.inference_mode()
def run_trt_inference(
    model, data: dict, engines: dict,
    max_gen: int = 256, num_traj_samples: int = 1, seed: int = 42,
) -> tuple[torch.Tensor, list[str], float]:
    """
    Run TRT plugin inference with batch decode.

    Returns: (pred_xyz [1, 1, num_traj_samples, T, 3], coc_texts, elapsed_ms)
    """
    from plugin_utils import create_kv_caches

    trt_vision, trt_lm, trt_diff = engines["trt_vision"], engines["trt_lm"], engines["trt_diffusion"]
    embed_tokens, lm_cfg = engines["embed_tokens"], engines["lm_config"]
    S, msl, nds = engines["S_input"], engines["max_seq_len"], engines["num_ds_layers"]
    rope_ref = engines["rope_deltas_ref"]
    hs = lm_cfg.hidden_size
    tokenizer = model.tokenizer
    eos_id = tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    traj_off = model.config.traj_token_start_idx
    traj_vs = model.config.traj_vocab_size
    B = num_traj_samples

    mi = prepare_model_inputs(model, data)
    input_ids, embeds, ds_embeds, vis_masks, _, _ = run_vlm_preprocessing(model, mi, trt_vision)

    # Build deepstack
    ds = torch.zeros(nds, B, msl, hs, dtype=FP16, device=DEVICE)
    vp = vis_masks[0].nonzero(as_tuple=True)[0]
    for i in range(nds):
        de = ds_embeds[i].to(FP16).squeeze(0)
        ds[i, :, vp, :] = de.unsqueeze(0).expand(B, -1, -1)

    start = time.perf_counter()
    if seed is not None:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    # Prefill (batch)
    e_batch = einops.repeat(embeds.to(FP16), "1 s h -> n s h", n=B)
    kvs = create_kv_caches(lm_cfg, msl, B, DEVICE, FP16)
    ctx = torch.tensor([S] * B, dtype=torch.int32, device=DEVICE)
    logits, _ = trt_lm(e_batch, kvs, ctx, ds)

    # Decode (batch)
    tokens = sample_token(logits, traj_off, traj_vs)
    gen_lists = [[t.item()] for t in tokens]
    pos = S
    seen_eos = tokens.squeeze(-1) == eos_id
    done = torch.zeros(B, dtype=torch.bool, device=DEVICE)

    for _ in range(max_gen - 1):
        if done.all():
            break
        se = embed_tokens(tokens.squeeze(-1)).to(FP16).unsqueeze(1)
        cs = torch.full((B,), pos + 1, dtype=torch.int32, device=DEVICE)
        logits, _ = trt_lm(se, kvs, cs, ds)
        tokens = sample_token(logits, traj_off, traj_vs)
        newly_done = seen_eos & ~done
        done = done | newly_done
        for i in range(B):
            if not done[i]:
                gen_lists[i].append(tokens[i].item())
        seen_eos = seen_eos | (tokens.squeeze(-1) == eos_id)
        pos += 1

    # CoC texts
    cocs = []
    for ids in gen_lists:
        txt = tokenizer.decode(ids, skip_special_tokens=False)
        cocs.append(txt.split(to_special_token("cot_end"))[0])

    # Build sequences + diffusion
    max_gl = max(len(g) for g in gen_lists)
    gen_t = torch.full((B, max_gl), tokenizer.pad_token_id, dtype=torch.long, device=DEVICE)
    for i, ids in enumerate(gen_lists):
        gen_t[i, :len(ids)] = torch.tensor(ids, device=DEVICE)
    id_batch = einops.repeat(input_ids.to(DEVICE), "1 s -> n s", n=B)
    full = replace_padding_after_eos(
        torch.cat([id_batch, gen_t], dim=1), eos_id, tokenizer.pad_token_id
    )

    tfs_mask = full == eos_id
    tfs_pos = torch.where(tfs_mask.any(1), tfs_mask.int().argmax(1),
                          torch.full((B,), full.shape[1]-1, device=DEVICE))
    offset = tfs_pos + 1

    nd = model.action_space.get_action_space_dims()[0]
    pid = einops.repeat(torch.arange(nd, device=DEVICE), "l -> 3 b l", b=B).clone()
    pid += (rope_ref.to(DEVICE) + offset[:, None]).to(pid.device)

    kv_len = pos
    pk = torch.stack([k[:, 0, :, :kv_len, :] for k in kvs])
    pv = torch.stack([k[:, 1, :, :kv_len, :] for k in kvs])
    neg = torch.finfo(FP16).min
    am = torch.zeros(B, 1, nd, kv_len + nd, dtype=FP16, device=DEVICE)
    for i in range(B):
        am[i, :, :, offset[i]:-nd] = neg

    def step_fn(x, t):
        return trt_diff(x.to(FP16), t.to(FP16), pk, pv, pid, am)

    action = model.diffusion.sample(batch_size=B, step_fn=step_fn, device=DEVICE, return_all_steps=False)

    hx = einops.repeat(data["ego_history_xyz"][:, -1].to(DEVICE), "1 ... -> n ...", n=B)
    hr = einops.repeat(data["ego_history_rot"][:, -1].to(DEVICE), "1 ... -> n ...", n=B)
    pred_xyz, _ = model.action_space.action_to_traj(action.float(), hx.float(), hr.float())
    pred_xyz = einops.rearrange(pred_xyz, "(b nj) ... -> b 1 nj ...", nj=B)

    elapsed = (time.perf_counter() - start) * 1000.0

    del kvs, pk, pv, ds
    torch.cuda.empty_cache()
    return pred_xyz, cocs, elapsed


# ============================================================
# Inference — PyTorch FP16
# ============================================================

@torch.inference_mode()
def run_pytorch_inference(
    model, data: dict, max_gen: int = 256, num_traj_samples: int = 1, seed: int = 42,
) -> tuple[torch.Tensor, list[str], float]:
    """
    Run PyTorch FP16 inference.

    Returns: (pred_xyz [1, 1, num_traj_samples, T, 3], coc_texts, elapsed_ms)
    """
    mi = prepare_model_inputs(model, data)
    if seed is not None:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    start = time.perf_counter()
    with torch.autocast("cuda", dtype=FP16):
        pred_xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=mi, top_p=0.98, temperature=0.6,
            num_traj_samples=num_traj_samples,
            max_generation_length=max_gen, return_extra=True,
        )
    elapsed = (time.perf_counter() - start) * 1000.0
    cocs = [extra["cot"][0][0][i] for i in range(num_traj_samples)]
    return pred_xyz, cocs, elapsed
