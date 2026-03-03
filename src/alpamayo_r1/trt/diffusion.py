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
TRT compilation of the Alpamayo diffusion denoising step.

The diffusion step (action_in_proj + expert + action_out_proj) is the hot path —
called N times per inference (default 10). We compile it as a single fused module
using torch.export + torch_tensorrt.dynamo.compile.

The VLM prefix KV context is provided as stacked tensors
[num_layers, B, num_kv_heads, prefix_len, head_dim] rather than a DynamicCache
object (which is not TRT-exportable). A PrefixKVCache is used inside
forward() so the expert can attend to the full VLM context.

Dynamic shapes
--------------
compile_diffusion_step_no_cache() uses a dynamic `prefix_len` dimension so a
single compiled engine can handle different VLM context lengths without
recompilation.  Provide `max_prefix_len` (upper bound) and an optional
`min_prefix_len` (default 1).  At runtime the `prefix_k`/`prefix_v` tensors
and `attention_mask` can have any prefix_len in [min_prefix_len, max_prefix_len].

Usage:
    from alpamayo_r1.trt.diffusion import compile_diffusion_step_no_cache

    trt_step = compile_diffusion_step_no_cache(model, max_prefix_len=4096,
                                               device="cuda")
"""

from __future__ import annotations

import logging
from contextlib import nullcontext

import torch
import torch.nn as nn

from alpamayo_r1.trt.prefix_cache import PrefixKVCache

logger = logging.getLogger(__name__)


class StaticKVDiffusionStepModule(nn.Module):
    """
    Fused action_in_proj + expert + action_out_proj for TRT compilation.

    The VLM prefix context is supplied as stacked KV tensors rather than a
    DynamicCache (not TRT-exportable). A PrefixKVCache is used inside
    forward() which concatenates prefix + new diffusion-token KV states via
    standard aten.cat.default (TRT-lowerable).

    Inputs
    ------
    x             : [B, *action_space_dims]
    t             : [B, 1, 1]
    prefix_k      : [num_layers, B, num_kv_heads, prefix_len, head_dim]
    prefix_v      : [num_layers, B, num_kv_heads, prefix_len, head_dim]
    position_ids  : [3, B, n_diffusion_tokens]
    attention_mask: [B, 1, n_diffusion_tokens, prefix_len + n_diffusion_tokens]
    """

    def __init__(
        self,
        action_in_proj: nn.Module,
        expert: nn.Module,
        action_out_proj: nn.Module,
        n_diffusion_tokens: int,
        action_space_dims: tuple[int, ...],
        num_layers: int,
    ):
        super().__init__()
        self.action_in_proj = action_in_proj
        self.expert = expert
        self.action_out_proj = action_out_proj
        self.n_diffusion_tokens = n_diffusion_tokens
        self.action_space_dims = action_space_dims
        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        future_token_embeds = self.action_in_proj(x, t)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(batch_size, self.n_diffusion_tokens, -1)

        past_key_values = PrefixKVCache(prefix_k, prefix_v)

        expert_out = self.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=False,
        )

        last_hidden = expert_out.last_hidden_state[:, -self.n_diffusion_tokens:]
        return self.action_out_proj(last_hidden).view(-1, *self.action_space_dims)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_diffusion_module(
    model: nn.Module,
    dtype: torch.dtype,
    device: str,
) -> tuple[StaticKVDiffusionStepModule, tuple, dict]:
    """
    Build the fused diffusion module and return it together with model-config
    values needed downstream.

    Returns (module, action_space_dims_tuple, cfg_dict) where cfg_dict has
    keys: n_diffusion_tokens, action_space_dims, num_layers, num_kv_heads, head_dim.
    """
    n_diffusion_tokens = model.action_space.get_action_space_dims()[0]
    action_space_dims = model.action_space.get_action_space_dims()
    expert_cfg = model.expert.config
    num_layers = expert_cfg.num_hidden_layers
    num_kv_heads = expert_cfg.num_key_value_heads
    head_dim = expert_cfg.head_dim

    model.expert.config._attn_implementation = "sdpa"

    module = (
        StaticKVDiffusionStepModule(
            action_in_proj=model.action_in_proj,
            expert=model.expert,
            action_out_proj=model.action_out_proj,
            n_diffusion_tokens=n_diffusion_tokens,
            action_space_dims=action_space_dims,
            num_layers=num_layers,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )

    cfg = dict(
        n_diffusion_tokens=n_diffusion_tokens,
        action_space_dims=action_space_dims,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    return module, cfg


def _make_sample_inputs(
    cfg: dict, min_prefix_len: int, max_prefix_len: int,
    dtype: torch.dtype, device: str,
) -> tuple[tuple, dict, callable]:
    """
    Build opt-profile sample inputs, dynamic shapes spec, and a helper that
    constructs inputs for any given prefix_len.

    Returns (sample_inputs, dynamic_shapes, make_inputs_fn).
    """
    n_diffusion_tokens = cfg["n_diffusion_tokens"]
    action_space_dims  = cfg["action_space_dims"]
    num_layers         = cfg["num_layers"]
    num_kv_heads       = cfg["num_kv_heads"]
    head_dim           = cfg["head_dim"]
    B = 1
    opt_prefix_len = (min_prefix_len + max_prefix_len) // 2

    def make_inputs(prefix_len: int) -> list[torch.Tensor]:
        return [
            torch.randn(B, *action_space_dims, dtype=dtype, device=device),
            torch.zeros(B, 1, 1, dtype=dtype, device=device),
            torch.zeros(num_layers, B, num_kv_heads, prefix_len, head_dim, dtype=dtype, device=device),
            torch.zeros(num_layers, B, num_kv_heads, prefix_len, head_dim, dtype=dtype, device=device),
            torch.arange(n_diffusion_tokens, device=device).unsqueeze(0).unsqueeze(0).expand(3, B, -1).clone(),
            torch.zeros(B, 1, n_diffusion_tokens, prefix_len + n_diffusion_tokens, dtype=dtype, device=device),
        ]

    sample_inputs = tuple(make_inputs(opt_prefix_len))

    prefix_dim = torch.export.Dim("prefix_len", min=min_prefix_len, max=max_prefix_len)
    mask_dim = prefix_dim + n_diffusion_tokens
    dynamic_shapes = (
        None,             # x: fully static
        None,             # t: fully static
        {3: prefix_dim},  # prefix_k: [num_layers, B, num_kv_heads, prefix_len, head_dim]
        {3: prefix_dim},  # prefix_v: same
        None,             # position_ids: static n_diffusion_tokens
        {3: mask_dim},    # attention_mask: [B, 1, n_diffusion_tokens, prefix_len + n_diffusion_tokens]
    )

    return sample_inputs, dynamic_shapes, make_inputs


def _export_diffusion_module(
    module: nn.Module,
    sample_inputs: tuple,
    dynamic_shapes: tuple,
) -> "torch.export.ExportedProgram":
    """Export with fallback to deferred-runtime-asserts on constraint violations."""
    with torch.no_grad():
        try:
            return torch.export.export(
                module, sample_inputs,
                dynamic_shapes=dynamic_shapes,
                strict=False,
            )
        except Exception as e:
            logger.warning(
                f"torch.export.export failed ({e}), "
                "retrying with prefer_deferred_runtime_asserts_over_guards=True..."
            )
            return torch.export._trace._export(
                module, sample_inputs,
                dynamic_shapes=dynamic_shapes,
                strict=False,
                prefer_deferred_runtime_asserts_over_guards=True,
            )


def _build_trt_input_specs(make_inputs: callable, min_prefix_len: int, opt_prefix_len: int, max_prefix_len: int):
    """Build min/opt/max TRT Input specs from the make_inputs factory."""
    import torch_tensorrt

    min_inputs = make_inputs(min_prefix_len)
    opt_inputs = make_inputs(opt_prefix_len)
    max_inputs = make_inputs(max_prefix_len)
    return [
        torch_tensorrt.Input(
            min_shape=t_min.shape,
            opt_shape=t_opt.shape,
            max_shape=t_max.shape,
            dtype=t_min.dtype,
        )
        for t_min, t_opt, t_max in zip(min_inputs, opt_inputs, max_inputs)
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compile_diffusion_step_no_cache(
    model: nn.Module,
    max_prefix_len: int,
    min_prefix_len: int = 1,
    device: str = "cuda",
    offload_module_to_cpu: bool = False,
    debug: bool = False,
    accuracy_check: bool = True,
) -> nn.Module:
    """
    Compile the fused diffusion step with a dynamic VLM KV prefix length.

    Uses torch.export + torch_tensorrt.dynamo.compile with a dynamic `prefix_len`
    dimension, so the single compiled engine handles any prefix length in
    [min_prefix_len, max_prefix_len] without recompilation.

    Args:
        model:          AlpamayoR1 model
        max_prefix_len: Upper bound for the VLM KV-cache prefix length (tokens).
        min_prefix_len: Lower bound for the prefix length (default 1).
        device:         CUDA device string
        offload_module_to_cpu:
                        Pass-through to torch_tensorrt.dynamo.compile
        debug:          Enable TRT debug logging
        accuracy_check: Compare TRT vs PyTorch on sample inputs after compilation

    Returns:
        TRT-compiled callable, also stored as model._trt_diffusion_step_no_cache
    """
    import torch_tensorrt

    dtype = torch.bfloat16
    module, cfg = _build_diffusion_module(model, dtype, device)

    logger.info("=" * 60)
    logger.info("Compiling Dynamic-KV Diffusion Step with TRT")
    logger.info(f"  prefix_len range:   [{min_prefix_len}, {max_prefix_len}]")
    logger.info(f"  n_diffusion_tokens: {cfg['n_diffusion_tokens']}")
    logger.info(f"  expert:             {cfg['num_layers']} layers, {cfg['num_kv_heads']} KV heads, {cfg['head_dim']} head_dim")
    logger.info(f"  action_space_dims:  {cfg['action_space_dims']}")
    logger.info(f"  offload_module_to_cpu: {offload_module_to_cpu}")

    sample_inputs, dynamic_shapes, make_inputs = _make_sample_inputs(
        cfg, min_prefix_len, max_prefix_len, dtype, device
    )

    ref_output = None
    if accuracy_check:
        with torch.no_grad():
            ref_output = module(*sample_inputs)
        logger.info(f"  PyTorch output shape: {ref_output.shape}")

    logger.info("Exporting with torch.export (dynamic prefix_len)...")
    exported = _export_diffusion_module(module, sample_inputs, dynamic_shapes)

    opt_prefix_len = (min_prefix_len + max_prefix_len) // 2
    trt_input_specs = _build_trt_input_specs(make_inputs, min_prefix_len, opt_prefix_len, max_prefix_len)

    trt_settings = {
        "use_explicit_typing": True,
        "use_fp32_acc": True,
        "truncate_double": True,
        "min_block_size": 1,
        "use_python_runtime": True,
        "debug": debug,
        "allow_complex_guards_as_runtime_asserts": True,
        "offload_module_to_cpu": offload_module_to_cpu,
    }

    logger.info("Compiling TRT engine (dynamic prefix_len)...")
    with torch_tensorrt.dynamo.Debugger() if debug else nullcontext():
        trt_step = torch_tensorrt.dynamo.compile(
            exported,
            inputs=trt_input_specs,
            **trt_settings,
        )

    if accuracy_check and ref_output is not None:
        with torch.no_grad():
            trt_output = trt_step(*sample_inputs)
        max_diff  = torch.abs(ref_output.float() - trt_output.float()).max().item()
        mean_diff = torch.abs(ref_output.float() - trt_output.float()).mean().item()
        logger.info(f"  max|Δ|  = {max_diff:.6f}")
        logger.info(f"  mean|Δ| = {mean_diff:.6f}")

    model._trt_diffusion_step_no_cache = trt_step
    logger.info("✓ Diffusion step compiled (stored as model._trt_diffusion_step_no_cache)")
    return trt_step


def save_diffusion_engine(
    model: nn.Module,
    path: str,
    max_prefix_len: int,
    min_prefix_len: int = 1,
    device: str = "cuda",
) -> bool:
    """
    Serialize the fused diffusion step as a raw ``.trt`` engine file (no
    torch_tensorrt required at inference time).

    Uses ``torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine``
    to produce raw TensorRT engine bytes and writes them to ``path``.  A JSON
    sidecar at ``<path>.json`` records shape metadata and the dynamic prefix_len
    range so ``TRTEngineRunner`` can be used without any Alpamayo model objects.

    Args:
        model:          AlpamayoR1 model
        path:           Destination ``.trt`` path
        max_prefix_len: Upper bound for VLM KV prefix length (tokens)
        min_prefix_len: Lower bound (default 1)
        device:         CUDA device string

    Returns:
        True on success, False on failure.
    """
    import torch_tensorrt

    from alpamayo_r1.trt.engine_io import save_trt_engine

    dtype = torch.bfloat16
    module, cfg = _build_diffusion_module(model, dtype, device)

    logger.info("=" * 60)
    logger.info("Serializing Dynamic-KV Diffusion Step as TRT engine")
    logger.info(f"  prefix_len range:   [{min_prefix_len}, {max_prefix_len}]")

    sample_inputs, dynamic_shapes, make_inputs = _make_sample_inputs(
        cfg, min_prefix_len, max_prefix_len, dtype, device
    )

    logger.info("Exporting with torch.export (for serialized engine)...")
    try:
        exported = _export_diffusion_module(module, sample_inputs, dynamic_shapes)
    except Exception as e:
        logger.error(f"All export methods failed: {e}")
        return False

    opt_prefix_len = (min_prefix_len + max_prefix_len) // 2
    trt_input_specs = _build_trt_input_specs(make_inputs, min_prefix_len, opt_prefix_len, max_prefix_len)

    trt_settings = {
        "truncate_double": True,
        "min_block_size": 1,
        "use_explicit_typing": True,
        "use_fp32_acc": True,
        "immutable_weights": True,
    }

    logger.info("Serializing diffusion engine with convert_exported_program_to_serialized_trt_engine...")
    try:
        engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
            exported,
            inputs=trt_input_specs,
            **trt_settings,
        )
    except Exception as e:
        logger.error(f"TRT serialization failed: {e}")
        return False

    metadata = {
        "component": "diffusion",
        "save_format": "raw_trt_engine",
        "precision": "BF16",
        "min_prefix_len": min_prefix_len,
        "max_prefix_len": max_prefix_len,
        "n_diffusion_tokens": cfg["n_diffusion_tokens"],
        "action_space_dims": list(cfg["action_space_dims"]),
        "num_layers": cfg["num_layers"],
        "num_kv_heads": cfg["num_kv_heads"],
        "head_dim": cfg["head_dim"],
    }
    save_trt_engine(engine_bytes, path, metadata)
    logger.info(f"✓ Diffusion engine saved to {path}")
    return True
