# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TRT compilation of the Qwen3VL vision encoder.

The vision model has data-dependent shapes (positional embeddings from grid_thw,
cu_seqlens for flash attention) that prevent direct torch.export. Workaround:

1. VisualFixedGrid wrapper — pre-computes all grid_thw-dependent values at init
   (pos_embeds, rotary cos/sin, cu_seqlens, static_lengths).
2. Patched attention forward — splits by static_lengths instead of cu_seqlens,
   enabling SDPA (TRT-compatible) instead of flash attention.
3. torch.export + torch_tensorrt.dynamo.compile on the wrapped model.

Why vision is static (not dynamic)
-----------------------------------
The vision encoder is intentionally compiled with **fully static shapes**.
The pos_embeds, rotary cos/sin, cu_seqlens, and attention static_lengths are all
derived from `grid_thw` at VisualFixedGrid.__init__ time and registered as
fixed buffers.  The total number of patch tokens (`total_patches`) is determined
by `grid_thw` and cannot vary independently — it equals sum(T*H*W) for each
image in the batch.

Adding a dynamic `total_patches` dimension would require:
  - Computing pos_embeds, cos/sin, cu_seqlens, and static_lengths at runtime
    (they are data-dependent functions of grid_thw)
  - Re-implementing the flash-attention split-by-sequence logic with a dynamic
    number of splits (not TRT-exportable)

Since Alpamayo uses a fixed camera rig with identical image resolution per
inference, grid_thw is constant across calls.  The per-call compile overhead
is therefore zero after the initial engine build.

Usage:
    from alpamayo_r1.trt.vision import compile_and_replace_vision_model

    compile_and_replace_vision_model(model, model_inputs)
    # model.vlm.model.visual.forward is now the TRT-compiled forward
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Store the original forward so the patch is idempotent
_orig_qwen3vl_attn_forward = None


def _patch_qwen3vl_vision_attention() -> bool:
    """
    Patch Qwen3VLVisionAttention.forward to use pre-computed static sequence
    lengths instead of cu_seqlens, enabling SDPA and TRT compilation.
    """
    global _orig_qwen3vl_attn_forward

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionAttention,
            apply_rotary_pos_emb_vision,
            eager_attention_forward,
        )
    except ImportError:
        logger.warning("Could not import Qwen3VL attention modules — patch not applied")
        return False

    if _orig_qwen3vl_attn_forward is not None:
        return True  # already patched

    _orig_qwen3vl_attn_forward = Qwen3VLVisionAttention.forward

    def _static_lengths_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        static_lengths = getattr(self, "_static_lengths", None)
        if static_lengths is None or self.config._attn_implementation == "flash_attention_2":
            return _orig_qwen3vl_attn_forward(
                self, hidden_states, cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # [1, num_heads, seq_len, head_dim]
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states   = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation in ALL_ATTENTION_FUNCTIONS:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        splits = [torch.split(t, static_lengths, dim=2)
                  for t in (query_states, key_states, value_states)]
        attn_outputs = [
            attention_interface(self, q, k, v, attention_mask=None,
                                scaling=self.scaling,
                                dropout=0.0 if not self.training else self.attention_dropout,
                                is_causal=False, **kwargs)[0]
            for q, k, v in zip(*splits)
        ]

        attn_output = torch.cat(attn_outputs, dim=1).reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)

    Qwen3VLVisionAttention.forward = _static_lengths_forward
    logger.info("✓ Patched Qwen3VLVisionAttention.forward for static lengths")
    return True


class VisualFixedGrid(nn.Module):
    """
    Wrapper for Qwen3VLVisionModel that pre-computes all grid_thw-dependent values.

    At init time we compute pos_embeds, rotary cos/sin, cu_seqlens, and
    static_lengths for a fixed grid_thw, then register them as buffers.
    The forward() ignores the grid_thw argument and uses the pre-computed buffers,
    making the entire graph statically shaped for torch.export.
    """

    def __init__(self, visual: nn.Module, grid_thw: torch.Tensor):
        super().__init__()
        self.visual = visual.eval()

        with torch.no_grad():
            pos_embeds = self.visual.fast_pos_embed_interpolate(grid_thw)

            rotary_pos_emb = self.visual.rot_pos_emb(grid_thw)
            seq_len = pos_embeds.shape[0]
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)

            cu_seqlens = torch.repeat_interleave(
                grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
            ).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

            static_lengths = (
                torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
                .cpu().tolist()
            )
            self._static_lengths = [int(x) for x in static_lengths]
            for blk in self.visual.blocks:
                blk.attn._static_lengths = self._static_lengths

        self.register_buffer("pos_embeds", pos_embeds, persistent=False)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)
        self.register_buffer("cu_seqlens", cu_seqlens, persistent=False)
        logger.info(f"VisualFixedGrid: {seq_len} tokens, static_lengths={self._static_lengths}")

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = self.visual.patch_embed(hidden_states)
        torch._check(hidden_states.shape[0] != 0)
        hidden_states = hidden_states + self.pos_embeds.to(hidden_states.dtype)

        position_embeddings = (
            self.cos.to(hidden_states.dtype),
            self.sin.to(hidden_states.dtype),
        )

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.visual.blocks):
            hidden_states = blk(hidden_states, cu_seqlens=self.cu_seqlens,
                                position_embeddings=position_embeddings)
            if layer_num in self.visual.deepstack_visual_indexes:
                idx = self.visual.deepstack_visual_indexes.index(layer_num)
                deepstack_feature_lists.append(
                    self.visual.deepstack_merger_list[idx](hidden_states)
                )

        hidden_states = self.visual.merger(hidden_states)
        return hidden_states, deepstack_feature_lists


class _PixelOnlyWrapper(nn.Module):
    """Single-input wrapper around VisualFixedGrid.

    The serialized engine only takes pixel_values; grid_thw is a
    VisualFixedGrid internal buffer so it does not need to be an engine input.
    """

    def __init__(self, vfg: VisualFixedGrid):
        super().__init__()
        self.vfg = vfg

    def forward(self, pixel_values: torch.Tensor):
        return self.vfg(pixel_values, grid_thw=None)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _prepare_vision_module(
    visual_model: nn.Module,
    model_inputs: dict[str, Any],
    device: str,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """
    Patch attention, configure the model, and return
    (VisualFixedGrid, pixel_values, image_grid_thw).
    """
    dtype = torch.bfloat16

    visual_model.config.attn_implementation = "sdpa"
    visual_model.config._attn_implementation = "sdpa"
    visual_model.config.use_cache = False
    visual_model = visual_model.to(dtype=dtype, device=device).eval()

    pixel_values   = model_inputs["tokenized_data"]["pixel_values"].to(dtype=dtype, device=device)
    image_grid_thw = model_inputs["tokenized_data"]["image_grid_thw"].to(device=device)

    wrapped = VisualFixedGrid(visual_model, image_grid_thw).to(device).eval()
    return wrapped, pixel_values, image_grid_thw


def _export_vision_module(
    module: nn.Module,
    inputs: tuple,
) -> "torch.export.ExportedProgram":
    """Export with fallback to deferred-runtime-asserts on constraint violations."""
    try:
        ep = torch.export.export(module, args=inputs, strict=False)
        logger.info("✓ Export succeeded")
        return ep
    except Exception as e:
        logger.warning(f"Standard export failed ({e}), trying _trace._export...")
        ep = torch.export._trace._export(
            module, args=inputs, strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )
        logger.info("✓ Trace export succeeded")
        return ep


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compile_vision_model(
    visual_model: nn.Module,
    model_inputs: dict[str, Any],
    device: str = "cuda",
    debug: bool = False,
    offload_module_to_cpu: bool = False,
) -> nn.Module | None:
    """
    Compile the Qwen3VL vision encoder with TRT.

    Args:
        visual_model: model.vlm.model.visual (Qwen3VLVisionModel)
        model_inputs: dict with tokenized_data containing pixel_values / image_grid_thw
        device:       CUDA device string
        debug:        Enable TRT debug logging

    Returns:
        TRT-compiled module, or None on failure
    """
    import torch_tensorrt

    if not _patch_qwen3vl_vision_attention():
        logger.error("Failed to patch vision attention — aborting")
        return None

    wrapped, pixel_values, image_grid_thw = _prepare_vision_module(
        visual_model, model_inputs, device
    )
    logger.info(f"  pixel_values shape: {pixel_values.shape}")
    logger.info(f"  image_grid_thw:     {image_grid_thw}")

    inputs = (pixel_values, None)

    trt_settings = {
        "truncate_double": True,
        "min_block_size": 1,
        "use_python_runtime": True,
        "immutable_weights": True,
        "offload_module_to_cpu": offload_module_to_cpu,
        "use_explicit_typing": True,
        "use_fp32_acc": True,
    }

    logger.info("Exporting with torch.export...")
    try:
        ep = _export_vision_module(wrapped, inputs)
    except Exception as e:
        logger.error(f"All export methods failed: {e}")
        return None

    logger.info("Compiling TRT engine...")
    try:
        with (
            torch_tensorrt.dynamo.Debugger(log_level="debug", engine_builder_monitor=False)
            if debug else nullcontext()
        ):
            trt_model = torch_tensorrt.dynamo.compile(ep, inputs, **trt_settings)
        logger.info("✓ TRT compilation succeeded")
    except Exception as e:
        logger.error(f"TRT compilation failed: {e}")
        return None

    # Accuracy check
    if offload_module_to_cpu:
        # TRT compile may offload underlying vision weights to CPU.
        # Move back before running the PyTorch reference check.
        wrapped = wrapped.to(device=device, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        torch_out = wrapped(*inputs)
        trt_out   = trt_model(*inputs)
    main_diff = torch.abs(torch_out[0].float() - trt_out[0].float())
    logger.info(f"  max|Δ|  = {main_diff.max().item():.6f}")
    logger.info(f"  mean|Δ| = {main_diff.mean().item():.6f}")

    return trt_model


def compile_and_replace_vision_model(
    model: nn.Module,
    model_inputs: dict[str, Any],
    device: str = "cuda",
    debug: bool = False,
    offload_module_to_cpu: bool = False,
) -> bool:
    """
    Compile the vision model and hot-swap it into the Alpamayo model.

    Replaces model.vlm.model.visual.forward with the TRT-compiled forward and
    stores the compiled module as model._trt_vision_model.

    Returns True on success, False on failure.
    """
    compiled = compile_vision_model(
        model.vlm.model.visual, model_inputs,
        device=device, debug=debug,
        offload_module_to_cpu=offload_module_to_cpu,
    )
    if compiled is None:
        logger.error("Vision model compilation failed")
        return False

    model.vlm.model.visual.forward = compiled.forward
    model._trt_vision_model = compiled
    logger.info("✓ Vision model replaced with TRT-compiled version")
    return True


def save_vision_engine(
    visual_model: nn.Module,
    model_inputs: dict[str, Any],
    path: str,
    device: str = "cuda",
    offload_module_to_cpu: bool = False,
) -> bool:
    """
    Serialize the vision encoder as a raw ``.trt`` engine file (no torch_tensorrt
    required at inference time).

    Uses ``torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine``
    to produce raw TensorRT engine bytes and writes them to ``path``.  A JSON
    sidecar at ``<path>.json`` records the pixel_values shape/dtype and the
    engine output names so ``TRTEngineRunner`` can be used without any
    Alpamayo model objects.

    Args:
        visual_model: ``model.vlm.model.visual`` (Qwen3VLVisionModel)
        model_inputs: dict with ``tokenized_data`` containing ``pixel_values``
                      and ``image_grid_thw``
        path:         Destination ``.trt`` path
        device:       CUDA device string

    Returns:
        True on success, False on failure.
    """
    import torch_tensorrt

    from alpamayo_r1.trt.engine_io import save_trt_engine

    if not _patch_qwen3vl_vision_attention():
        logger.error("Failed to patch vision attention — aborting")
        return False

    wrapped_base, pixel_values, image_grid_thw = _prepare_vision_module(
        visual_model, model_inputs, device
    )

    # Wrap in _PixelOnlyWrapper so the serialized engine takes only pixel_values
    # (grid_thw is a VisualFixedGrid internal buffer, not an engine input).
    wrapped = _PixelOnlyWrapper(wrapped_base).to(device).eval()
    inputs  = (pixel_values,)

    # use_explicit_typing=True infers precision from the model's dtypes — do not
    # also set enabled_precisions (they are mutually exclusive).
    trt_settings = {
        "truncate_double": True,
        "min_block_size": 1,
        "use_explicit_typing": True,
        "use_fp32_acc": True,
        "immutable_weights": True,
        "offload_module_to_cpu": offload_module_to_cpu,
    }

    logger.info("Exporting with torch.export (for serialized engine)...")
    try:
        ep = _export_vision_module(wrapped, inputs)
    except Exception as e:
        logger.error(f"All export methods failed: {e}")
        return False

    trt_input_spec = torch_tensorrt.Input.from_tensor(pixel_values)

    logger.info("Serializing TRT engine (vision)...")
    try:
        engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
            ep,
            inputs=(trt_input_spec,),
            **trt_settings,
        )
    except Exception as e:
        logger.error(f"TRT serialization failed: {e}")
        return False

    metadata = {
        "component": "vision",
        "precision": "BF16",
        "pixel_values_shape": list(pixel_values.shape),
        "pixel_values_dtype": str(pixel_values.dtype),
        "image_grid_thw": image_grid_thw.tolist(),
    }
    save_trt_engine(engine_bytes, path, metadata)
    logger.info(f"✓ Vision engine saved to {path}")
    return True
