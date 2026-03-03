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
TRT LM compilation with explicit KV-cache I/O.

This module provides a wrapper around Qwen3VLTextModel that:
  - accepts KV cache in forward as stacked tensors (prefix_k, prefix_v)
  - returns updated stacked KV tensors (updated_k, updated_v)

Tensor shapes:
  prefix_k/prefix_v: [num_layers, batch, num_kv_heads, prefix_len, head_dim]
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from types import MethodType

import torch
import torch.nn as nn

from alpamayo_r1.trt.prefix_cache import (
    PrefixKVCache,
    extract_stacked_kv_from_cache,
    maybe_to as _maybe_to,
)

logger = logging.getLogger(__name__)


class Qwen3VLTextModelWithCacheWrapper(nn.Module):
    """
    Wrapper around Qwen3VLTextModel for torch.export/TRT with explicit KV cache tensors.
    """

    def __init__(self, language_model: nn.Module):
        super().__init__()
        self.language_model = language_model

    @staticmethod
    def _build_causal_mask(
        batch_size: int,
        q_len: int,
        prefix_len: int,
        attention_mask: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        neg_inf = torch.finfo(torch.float32).min
        kv_len = prefix_len + q_len

        # Fast path for single-token decode.
        if q_len == 1:
            causal_mask = torch.zeros(batch_size, 1, 1, kv_len, device=device, dtype=torch.float32)
            if attention_mask is not None:
                if attention_mask.ndim == 4:
                    keep = attention_mask[:, :, -1:, :].to(torch.bool)
                else:
                    keep = attention_mask[:, None, None, :].to(torch.bool)
                causal_mask = torch.where(
                    keep,
                    causal_mask,
                    torch.full((), neg_inf, dtype=torch.float32, device=device),
                )
            return causal_mask.to(dtype=dtype)

        future = torch.triu(torch.ones(q_len, q_len, device=device, dtype=torch.bool), diagonal=1)
        causal_q = torch.zeros(q_len, q_len, device=device, dtype=torch.float32).masked_fill(future, neg_inf)
        base = torch.cat(
            [
                torch.zeros(q_len, prefix_len, device=device, dtype=torch.float32),
                causal_q,
            ],
            dim=-1,
        )
        causal_mask = base.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, q_len, kv_len)
        if attention_mask is not None:
            keep = attention_mask[:, None, None, :].to(torch.bool)
            causal_mask = causal_mask.masked_fill(~keep, neg_inf)
        return causal_mask.to(dtype=dtype)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,  # PrefixKVCache or (prefix_k, prefix_v)
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
    ):
        del use_cache, visual_pos_masks, deepstack_visual_embeds

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        return_tensor_kv = isinstance(past_key_values, tuple)

        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)
        bsz = inputs_embeds.shape[0]

        if isinstance(past_key_values, PrefixKVCache):
            cache = past_key_values
        elif past_key_values is None:
            num_layers = self.language_model.config.num_hidden_layers
            num_kv_heads = self.language_model.config.num_key_value_heads
            head_dim = self.language_model.config.head_dim
            cache = PrefixKVCache.empty(
                num_layers=num_layers,
                batch_size=bsz,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
        elif isinstance(past_key_values, tuple) and len(past_key_values) == 2:
            cache = PrefixKVCache(
                _maybe_to(
                    past_key_values[0],
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                ),
                _maybe_to(
                    past_key_values[1],
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                ),
            )
        else:
            raise ValueError("Wrapper expects past_key_values=PrefixKVCache or (prefix_k, prefix_v)")

        if cache_position is None:
            prefix_len = cache.get_seq_length()
            cache_position = torch.arange(
                prefix_len,
                prefix_len + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        text_position_ids = position_ids[0]
        if (
            attention_mask is not None
            and attention_mask.ndim == 4
            and attention_mask.shape[-2] == inputs_embeds.shape[1]
        ):
            expected_kv = cache.get_seq_length() + inputs_embeds.shape[1]
            if attention_mask.shape[-1] != expected_kv:
                if attention_mask.shape[-1] > expected_kv:
                    attention_mask = attention_mask[..., -expected_kv:]
                else:
                    pad = torch.zeros(
                        attention_mask.shape[0],
                        attention_mask.shape[1],
                        attention_mask.shape[2],
                        expected_kv - attention_mask.shape[-1],
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat([pad, attention_mask], dim=-1)
            causal_mask = attention_mask.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        else:
            causal_mask = self._build_causal_mask(
                batch_size=inputs_embeds.shape[0],
                q_len=inputs_embeds.shape[1],
                prefix_len=cache.get_seq_length(),
                attention_mask=attention_mask,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )

        hidden_states = inputs_embeds
        position_embeddings = self.language_model.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.language_model.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.language_model.norm(hidden_states)
        updated_k, updated_v = cache.get_updated_stacked()
        if return_tensor_kv:
            return hidden_states, updated_k, updated_v
        if (updated_k is not cache.key_cache) or (updated_v is not cache.value_cache):
            cache.update_stacked(updated_k, updated_v)
        return hidden_states, cache


def _normalize_attention_mask_for_wrapper(
    attention_mask: torch.Tensor | dict | None,
    batch_size: int,
    query_len: int,
    prefix_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert incoming attention mask variants into [B, prefix_len + query_len] with 1=keep."""
    target_len = prefix_len + query_len
    if isinstance(attention_mask, dict):
        attention_mask = attention_mask.get("full_attention")

    if attention_mask is None:
        return torch.ones(batch_size, target_len, dtype=torch.long, device=device)

    mask = attention_mask if attention_mask.device == device else attention_mask.to(device=device)
    if (
        mask.ndim == 2
        and not mask.dtype.is_floating_point
        and mask.shape[1] == target_len
        and mask.dtype in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    ):
        return mask

    if mask.ndim == 4:
        key_row = mask[:, 0, -1, :]
        if key_row.dtype.is_floating_point:
            thresh = torch.finfo(key_row.dtype).min / 2
            mask2d = (key_row > thresh).to(torch.long)
        else:
            mask2d = key_row.to(torch.bool).to(torch.long)
    elif mask.ndim == 2:
        if mask.dtype.is_floating_point:
            mask2d = (mask > 0).to(torch.long)
        else:
            mask2d = mask.to(torch.bool).to(torch.long)
    else:
        raise ValueError(f"Unsupported attention_mask rank for wrapper LM: ndim={mask.ndim}")

    cur_len = mask2d.shape[1]
    if cur_len == query_len and prefix_len > 0:
        pad = torch.ones(batch_size, prefix_len, dtype=torch.long, device=device)
        mask2d = torch.cat([pad, mask2d], dim=-1)
    elif cur_len < target_len:
        pad = torch.ones(batch_size, target_len - cur_len, dtype=torch.long, device=device)
        mask2d = torch.cat([pad, mask2d], dim=-1)
    elif cur_len > target_len:
        mask2d = mask2d[:, -target_len:]

    return mask2d


def _export_wrapper(
    wrapper: nn.Module,
    example_attention_mask: torch.Tensor,
    example_position_ids: torch.Tensor,
    example_embeds: torch.Tensor,
    example_past_key_values: tuple[torch.Tensor, torch.Tensor],
    max_batch_size: int,
    max_seq_len: int,
    max_prefix_len: int,
) -> "torch.export.ExportedProgram":
    import transformers.integrations.sdpa_attention as _sdpa_mod

    orig_use_gqa = _sdpa_mod.use_gqa_in_sdpa
    _sdpa_mod.use_gqa_in_sdpa = lambda *args, **kwargs: False

    batch_dim = torch.export.Dim("batch", min=1, max=max_batch_size)
    seq_dim = torch.export.Dim("seq_len", min=1, max=max_seq_len)
    prefix_dim = torch.export.Dim("prefix_len", min=0, max=max_prefix_len)
    mask_dim = torch.export.Dim("mask_len", min=1, max=max_prefix_len + max_seq_len)
    dynamic_shapes = {
        "attention_mask": {0: batch_dim, 1: mask_dim},
        "position_ids": {1: batch_dim, 2: seq_dim},
        "inputs_embeds": {0: batch_dim, 1: seq_dim},
        "past_key_values": (
            {1: batch_dim, 3: prefix_dim},
            {1: batch_dim, 3: prefix_dim},
        ),
    }

    try:
        with torch.no_grad():
            kwargs = dict(
                attention_mask=example_attention_mask,
                position_ids=example_position_ids,
                inputs_embeds=example_embeds,
                past_key_values=example_past_key_values,
            )
            try:
                ep = torch.export.export(
                    wrapper,
                    args=(),
                    kwargs=kwargs,
                    dynamic_shapes=dynamic_shapes,
                    strict=False,
                )
            except Exception as e:
                logger.warning("torch.export.export failed (%s); trying trace fallback", e)
                ep = torch.export._trace._export(
                    wrapper,
                    args=(),
                    kwargs=kwargs,
                    dynamic_shapes=dynamic_shapes,
                    strict=False,
                    prefer_deferred_runtime_asserts_over_guards=True,
                )
    finally:
        _sdpa_mod.use_gqa_in_sdpa = orig_use_gqa

    return ep


def _fix_requires_output_allocator(trt_backbone: nn.Module) -> None:
    try:
        from torch_tensorrt.dynamo.runtime._PythonTorchTensorRTModule import (
            PythonTorchTensorRTModule,
        )
    except ImportError:
        logger.warning("Could not import PythonTorchTensorRTModule; skipping OA fix")
        return

    fixed = 0
    for mod in trt_backbone.modules():
        if isinstance(mod, PythonTorchTensorRTModule) and not mod.requires_output_allocator:
            mod.requires_output_allocator = True
            mod.create_output_allocator()
            fixed += 1
    logger.info("Enabled output allocator on %d TRT submodule(s)", fixed)


def _install_hf_generate_wrapper_adapter(
    model: nn.Module,
    wrapper: nn.Module,
    *,
    device: str,
    dtype: torch.dtype,
) -> None:
    """Patch language_model.forward and bridge HF cache <-> PrefixKVCache."""
    from transformers.modeling_outputs import BaseModelOutputWithPast

    language_model = model.vlm.model.language_model
    wrapper_device = torch.device(device)

    if not hasattr(language_model, "_wrapper_original_forward"):
        language_model._wrapper_original_forward = language_model.forward

    language_model._wrapper_impl = wrapper
    language_model._wrapper_dtype = dtype
    language_model._wrapper_device = wrapper_device

    def _wrapper_forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        **kwargs,
    ):
        def _call_original_forward():
            return self._wrapper_original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                **kwargs,
            )

        if use_cache is False:
            return _call_original_forward()

        # Preserve correctness for DeepStack visual-prefill path.
        if visual_pos_masks is not None or deepstack_visual_embeds is not None:
            return _call_original_forward()

        del kwargs
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = _maybe_to(
            inputs_embeds,
            device=self._wrapper_device,
            dtype=self._wrapper_dtype,
        )
        if position_ids is not None:
            position_ids = _maybe_to(position_ids, device=self._wrapper_device)
        if cache_position is not None:
            cache_position = _maybe_to(cache_position, device=self._wrapper_device)

        bsz, q_len = inputs_embeds.shape[:2]
        prefix_k, prefix_v = extract_stacked_kv_from_cache(
            past_key_values,
            num_layers=self.config.num_hidden_layers,
            batch_size=bsz,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            device=self._wrapper_device,
            dtype=self._wrapper_dtype,
        )
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
            # Fast path for HF decode: keep 2D mask in its native integer/bool dtype.
            wrapper_attention_mask = _maybe_to(attention_mask, device=self._wrapper_device)
            target_len = prefix_k.shape[3] + q_len
            cur_len = wrapper_attention_mask.shape[1]
            if cur_len > target_len:
                wrapper_attention_mask = wrapper_attention_mask[:, -target_len:]
            elif cur_len < target_len:
                pad = torch.ones(
                    bsz,
                    target_len - cur_len,
                    dtype=wrapper_attention_mask.dtype,
                    device=self._wrapper_device,
                )
                wrapper_attention_mask = torch.cat([pad, wrapper_attention_mask], dim=-1)
        else:
            wrapper_attention_mask = _normalize_attention_mask_for_wrapper(
                attention_mask=attention_mask,
                batch_size=bsz,
                query_len=q_len,
                prefix_len=prefix_k.shape[3],
                device=self._wrapper_device,
            )

        # TRT-exported wrapper requires concrete position_ids on every call.
        if cache_position is None:
            prefix_len = prefix_k.shape[3]
            cache_position = torch.arange(
                prefix_len,
                prefix_len + q_len,
                device=self._wrapper_device,
                dtype=torch.long,
            )
        elif cache_position.ndim == 0:
            cache_position = cache_position.view(1)
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, bsz, -1).clone()
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, bsz, -1).clone()
        position_ids = _maybe_to(position_ids, device=self._wrapper_device).to(dtype=torch.long)

        with torch.no_grad():
            # Compiled FX/TRT path: exported graph takes positional tensor inputs.
            wrapper_out = self._wrapper_impl(
                wrapper_attention_mask,
                position_ids,
                inputs_embeds,
                (prefix_k, prefix_v),
            )

        if not isinstance(wrapper_out, (tuple, list)):
            raise TypeError(f"Wrapper output must be tuple/list, got {type(wrapper_out)}")
        if len(wrapper_out) == 3:
            hidden_states, updated_k, updated_v = wrapper_out
            updated_prefix_cache = PrefixKVCache(updated_k, updated_v)
        else:
            raise ValueError(
                "Compiled wrapper is expected to return 3 outputs "
                f"(hidden_states, updated_k, updated_v); got {len(wrapper_out)}"
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=updated_prefix_cache,
        )

    language_model.forward = MethodType(_wrapper_forward, language_model)


def compile_vlm_lm_trt_with_cache(
    model: nn.Module,
    max_seq_len: int = 4096,
    max_prefix_len: int = 4096,
    batch_size: int = 2,
    device: str = "cuda",
    offload_module_to_cpu: bool = False,
    debug: bool = False,
    accuracy_check: bool = False,
) -> nn.Module:
    """
    Compile Qwen3-VL language model with explicit KV-cache tensor I/O.

    The exported TRT graph supports dynamic batch in [1, batch_size].
    """
    import torch_tensorrt

    logger.info("=" * 65)
    logger.info("Compiling VLM LM (KV cache tensor I/O) with TRT")
    logger.info("=" * 65)
    logger.info("  max_seq_len: %d", max_seq_len)
    logger.info("  max_prefix_len: %d", max_prefix_len)
    logger.info("  batch_size: %d", batch_size)
    logger.info("  offload_module_to_cpu: %s", offload_module_to_cpu)

    dtype = torch.bfloat16

    backbone = model.vlm.model
    language_model = backbone.language_model

    hidden_size = language_model.config.hidden_size
    num_layers = language_model.config.num_hidden_layers
    num_kv_heads = language_model.config.num_key_value_heads
    head_dim = language_model.config.head_dim

    language_model.config._attn_implementation = "sdpa"
    for layer in language_model.layers:
        if hasattr(layer.self_attn, "_attn_implementation"):
            layer.self_attn._attn_implementation = "sdpa"
        if hasattr(layer.self_attn, "config"):
            layer.self_attn.config._attn_implementation = "sdpa"

    wrapper = Qwen3VLTextModelWithCacheWrapper(language_model).to(device=device, dtype=dtype).eval()

    bsz = int(batch_size)
    if bsz <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    opt_seq_len = min(max_seq_len, 128)
    opt_prefix_len = min(max_prefix_len, 256)
    example_embeds = torch.randn(bsz, opt_seq_len, hidden_size, dtype=dtype, device=device)
    example_prefix_k = torch.zeros(
        num_layers, bsz, num_kv_heads, opt_prefix_len, head_dim, dtype=dtype, device=device
    )
    example_prefix_v = torch.zeros_like(example_prefix_k)
    example_past_key_values = (example_prefix_k, example_prefix_v)
    example_attention_mask = torch.ones(
        bsz, opt_prefix_len + opt_seq_len, dtype=torch.long, device=device
    )
    example_position_ids = torch.arange(opt_seq_len, device=device).view(1, 1, -1).expand(3, bsz, -1).clone()

    ep = _export_wrapper(
        wrapper=wrapper,
        example_attention_mask=example_attention_mask,
        example_position_ids=example_position_ids,
        example_embeds=example_embeds,
        example_past_key_values=example_past_key_values,
        max_batch_size=bsz,
        max_seq_len=max_seq_len,
        max_prefix_len=max_prefix_len,
    )

    trt_settings: dict = {
        "use_explicit_typing": True,
        "use_fp32_acc": True,
        "truncate_double": True,
        "min_block_size": 1,
        "use_python_runtime": True,
        "debug": debug,
        "allow_complex_guards_as_runtime_asserts": True,
        "offload_module_to_cpu": False,
    }

    trt_inputs = [
        example_attention_mask,
        example_position_ids,
        example_embeds,
        example_past_key_values,
    ]

    with torch_tensorrt.dynamo.Debugger() if debug else nullcontext():
        trt_backbone = torch_tensorrt.dynamo.compile(
            ep,
            inputs=trt_inputs,
            **trt_settings,
        )
    _fix_requires_output_allocator(trt_backbone)

    check_len = min(max_seq_len, 64)
    check_prefix_len = min(max_prefix_len, 32)
    short_embeds = torch.randn(1, check_len, hidden_size, dtype=dtype, device=device)
    short_attn = torch.ones(1, check_prefix_len + check_len, dtype=torch.long, device=device)
    short_pos = torch.arange(check_len, device=device).view(1, 1, -1).expand(3, 1, -1).clone()
    short_prefix_k = torch.zeros(
        num_layers, 1, num_kv_heads, check_prefix_len, head_dim, dtype=dtype, device=device
    )
    short_prefix_v = torch.zeros_like(short_prefix_k)
    try:
        with torch.no_grad():
            trt_out_raw = trt_backbone(
                short_attn,
                short_pos,
                short_embeds,
                (short_prefix_k, short_prefix_v),
            )
            if accuracy_check:
                ref_h = wrapper(
                    attention_mask=short_attn,
                    position_ids=short_pos,
                    inputs_embeds=short_embeds,
                    past_key_values=(short_prefix_k, short_prefix_v),
                )[0]
                trt_h = trt_out_raw[0]
                l2_err = (ref_h.float() - trt_h.float()).norm() / ref_h.float().norm()
                logger.info("LM TRT smoke-check L2 relative error: %.4f", l2_err)
    except Exception as e:
        logger.warning("LM TRT smoke test failed: %s", e)

    model._trt_vlm_backbone = trt_backbone
    logger.info("✓ VLM LM compiled with KV cache I/O (stored as model._trt_vlm_backbone)")

    _install_hf_generate_wrapper_adapter(
        model=model,
        wrapper=trt_backbone,
        device=device,
        dtype=dtype,
    )
    logger.info("Installed trt_backbone adapter on language_model.forward")
    return trt_backbone
