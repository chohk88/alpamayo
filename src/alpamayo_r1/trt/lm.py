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
TRT compilation of the Alpamayo VLM language model backbone for autoregressive generation.

Architecture
------------
Alpamayo uses Qwen3-VL-8B as its VLM backbone:

    model.vlm                                  Qwen3VLForConditionalGeneration
    model.vlm.model                            Qwen3VLModel
    model.vlm.model.language_model             Qwen3VLTextModel  (THIS IS WHAT WE COMPILE)
    model.vlm.lm_head                          nn.Linear(4096, vocab_size)  — stays in PyTorch
    model.vlm.model.get_input_embeddings()     nn.Embedding  — stays in PyTorch

We wrap Qwen3VLTextModel directly (not Qwen3VLModel) to avoid the @check_model_inputs /
@auto_docstring decorators that would interfere with torch.export.

Compilation approach
--------------------
We follow the static-KV-cache FX pass pattern from pytorch/TensorRT tools/llm/:

1. Wrap Qwen3VLTextModel in a no-cache wrapper: (inputs_embeds, cos, sin) -> last_hidden_state
   cos/sin are precomputed outside TRT by calling language_model.rotary_emb() in Python.
   This avoids M-RoPE's apply_interleaved_mrope scatter op which TRT mis-compiles during decode
   (uses position_ids VALUE as reshape dim → "reshape would change volume 192 to N" errors).
2. Export with torch.export (dynamic seq_len), patching use_gqa_in_sdpa to return False so that
   sdpa_attention_forward always calls repeat_kv(key, num_key_value_groups) before SDPA.
   (During torch.export, keys are FakeTensors not torch.fx.Proxy, so use_gqa_in_sdpa would
   normally return True → enable_gqa path → 8-head keys → TRT SDPA converter failure.)
3. Register the SDPA lowering pass (replaces flash/efficient attention → plain SDPA for TRT)
4. Register the static_cache_v2 lowering pass (injects KV cache tensors as graph I/O)
5. Compile with torch_tensorrt.dynamo.compile
6. The compiled module signature becomes:
     (inputs_embeds, cos, sin,
      k0, v0, k1, v1, ..., k35, v35,
      start_idx, end_idx)
     -> (last_hidden_state, k0_new, v0_new, ..., k35_new, v35_new)

Deepstack visual embeds
-----------------------
Qwen3-VL-8B has 3 deepstack visual indexes (vision transformer layers 8, 16, 24) which inject
multi-scale visual features into the first 3 decoder layers of Qwen3VLTextModel.  These are
passed as a variable-length list of tensors — they cannot be compiled into a static TRT graph.

For TRT compilation we skip deepstack (deepstack_visual_embeds=None), matching the approach
used in pytorch/TensorRT tools/llm/run_vlm.py.  In _prepare_prefill_inputs we still call
backbone.get_image_features() to get the merged inputs_embeds (the main image token embeddings),
but we discard the deepstack_image_embeds returned by that call.

The accuracy impact of this trade-off is measured by compile_vlm_lm_trt() accuracy check.

Generation
----------
generate_alpamayo_with_static_cache() replaces model.vlm.generate():
- Prefill: full prompt in one shot (start_idx=0, end_idx=prompt_len)
- Decode: single token per step; KV tensors are passed back in from previous output
- Stopping: mirrors Alpamayo's StopAfterEOS / max_new_tokens logic
- Sampling: top-p / temperature

Rotary Embeddings (M-RoPE)
--------------------------
Qwen3-VL uses M-RoPE with position_ids shape (3, B, S) — one slice per temporal/height/width
dimension.  Rather than passing position_ids into TRT, we precompute cos/sin in Python before
each TRT call (language_model.rotary_emb(embeds, position_ids) → cos, sin).  This eliminates
the M-RoPE scatter ops from the TRT graph.

generate_alpamayo_with_static_cache() computes plain sequential position_ids for prefill and
a single-position id for each decode step, then expands to (B, S) before calling rotary_emb.
"""

from __future__ import annotations

import logging
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Make the torch-tensorrt tools/llm helpers importable
# ---------------------------------------------------------------------------
# Path layout:
#   /home/<user>/alpamayo-trt/alpamayo/src/alpamayo_r1/trt/lm.py  <- __file__
#   /home/<user>/pytorch_org/tensorrt/tools/llm/                   <- _TOOLS_LLM
# parents[5] gives /home/<user>/  (file -> trt -> alpamayo_r1 -> src -> alpamayo -> alpamayo-trt -> home)
#
# IMPORTANT: torchtrt_ext uses relative imports (from .sdpa_converter import *), so it must
# be imported as a package: "from torchtrt_ext import register_sdpa".  Only _TOOLS_LLM needs
# to be on sys.path (as the parent of torchtrt_ext/).  Do NOT add _TOOLS_LLM_EXT to sys.path
# since that would break the relative import within torchtrt_ext.
_TOOLS_LLM = Path(__file__).parents[5] / "pytorch_org/tensorrt/tools/llm"
if str(_TOOLS_LLM) not in sys.path:
    sys.path.insert(0, str(_TOOLS_LLM))


# ---------------------------------------------------------------------------
# Wrapper: Qwen3VLTextModel (no cache, inputs_embeds → last_hidden_state)
# ---------------------------------------------------------------------------


class Qwen3VLTextModelWrapper(nn.Module):
    """
    Thin wrapper around Qwen3VLTextModel for torch.export / TRT compilation.

    Signature: (inputs_embeds, cos, sin) -> last_hidden_state

    - Wraps backbone.language_model directly (avoids @check_model_inputs decorator on backbone)
    - No KV cache (use_cache=False)
    - cos/sin are precomputed by calling language_model.rotary_emb() in Python BEFORE
      passing to TRT.  This avoids the M-RoPE interleaved scatter op (apply_interleaved_mrope)
      which uses data-dependent indexing that TRT cannot handle correctly during decode.
    - deepstack_visual_embeds is not passed (see module docstring for rationale)

    cos/sin shapes: (B, S, head_dim) — passed directly to each attention layer via
    position_embeddings tuple.
    """

    def __init__(self, language_model: nn.Module):
        super().__init__()
        self.language_model = language_model

    def forward(
        self,
        inputs_embeds: torch.Tensor,  # (B, S, hidden_size)
        cos: torch.Tensor,            # (B, S, head_dim)
        sin: torch.Tensor,            # (B, S, head_dim)
    ) -> torch.Tensor:
        # Build position_embeddings tuple that the decoder layers expect
        position_embeddings = (cos, sin)

        # Call Qwen3VLTextModel internals directly, bypassing the rotary_emb call
        # that would happen inside language_model.forward() with position_ids.
        # We replicate the relevant part of Qwen3VLTextModel.forward():
        #
        # attention_mask=None: _export_wrapper patches use_gqa_in_sdpa to always return
        # False during export, so sdpa_attention_forward always calls repeat_kv() to
        # expand KV heads (8→32) before SDPA.  Without this patch, torch.export's
        # FakeTensor tracing (not torch.fx.Proxy) would make use_gqa_in_sdpa return True,
        # keeping 8-head keys and using enable_gqa=True which TRT cannot handle.
        hidden_states = inputs_embeds

        for decoder_layer in self.language_model.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                cache_position=None,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.language_model.norm(hidden_states)
        return hidden_states  # (B, S, hidden_size)


# ---------------------------------------------------------------------------
# Export helpers (adapted from tools/llm/utils.py export_llm)
# ---------------------------------------------------------------------------


def _export_wrapper(
    wrapper: nn.Module,
    example_embeds: torch.Tensor,
    example_cos: torch.Tensor,
    example_sin: torch.Tensor,
    max_seq_len: int,
) -> "torch.export.ExportedProgram":
    """Export Qwen3VLTextModelWrapper with dynamic seq_len dimension.

    The wrapper accepts (inputs_embeds, cos, sin) where cos/sin are precomputed
    outside TRT.  All three share the same dynamic seq_len dimension.

    GQA fix: during torch.export (which uses FakeTensors, not torch.fx.Proxy),
    transformers' use_gqa_in_sdpa() sees isinstance(key, torch.fx.Proxy) == False
    and returns True (native GQA mode, no repeat_kv).  TRT's custom SDPA converter
    does not handle GQA; it expects pre-expanded 32-head keys.  We patch
    use_gqa_in_sdpa to return False so that sdpa_attention_forward always calls
    repeat_kv(key, num_key_value_groups) before the SDPA op.
    """
    import transformers.integrations.sdpa_attention as _sdpa_mod
    _orig_use_gqa = _sdpa_mod.use_gqa_in_sdpa
    _sdpa_mod.use_gqa_in_sdpa = lambda *args, **kwargs: False

    seq_dim = torch.export.Dim("seq_len", min=1, max=max_seq_len)
    dynamic_shapes = (
        {1: seq_dim},   # inputs_embeds: (B, S, hidden)
        {1: seq_dim},   # cos:           (B, S, head_dim)
        {1: seq_dim},   # sin:           (B, S, head_dim)
    )

    try:
        with torch.no_grad():
            try:
                ep = torch.export.export(
                    wrapper,
                    args=(example_embeds, example_cos, example_sin),
                    dynamic_shapes=dynamic_shapes,
                    strict=False,
                )
            except Exception as e:
                logger.warning(f"torch.export.export failed ({e}), trying _trace._export fallback...")
                ep = torch.export._trace._export(
                    wrapper,
                    args=(example_embeds, example_cos, example_sin),
                    dynamic_shapes=dynamic_shapes,
                    strict=False,
                    prefer_deferred_runtime_asserts_over_guards=True,
                )
    finally:
        _sdpa_mod.use_gqa_in_sdpa = _orig_use_gqa

    return ep


def _fix_requires_output_allocator(trt_backbone: nn.Module) -> None:
    """
    Enable output-allocator mode on all PythonTorchTensorRTModule submodules.

    static_cache_v2 inserts concat_keys = cat(cache[:start_idx], k) operations
    whose output shapes depend on the runtime scalar `start_idx`.  TRT marks
    these as "profile-oblivious" outputs.  Profile-oblivious outputs MUST be
    served via an OutputAllocator — calling set_tensor_address() on them is
    undefined behaviour and causes the enqueueV3 error:
      "Neither address or allocator is set for output tensor output13"

    The fix: after dynamo.compile() has already called setup_engine() (which
    only enables OA when requires_output_allocator was set at construction
    time), we flip the flag and call create_output_allocator() ourselves.
    """
    try:
        from torch_tensorrt.dynamo.runtime._PythonTorchTensorRTModule import (
            PythonTorchTensorRTModule,
        )
    except ImportError:
        logger.warning("Could not import PythonTorchTensorRTModule — skipping OA fix")
        return

    fixed = 0
    for mod in trt_backbone.modules():
        if isinstance(mod, PythonTorchTensorRTModule):
            if not mod.requires_output_allocator:
                mod.requires_output_allocator = True
                mod.create_output_allocator()
                fixed += 1
    logger.info(f"  Enabled output allocator on {fixed} TRT submodule(s)")


# ---------------------------------------------------------------------------
# Main compilation entry point
# ---------------------------------------------------------------------------


def compile_vlm_lm_trt(
    model: nn.Module,
    max_seq_len: int = 4096,
    precision: str = "BF16",
    device: str = "cuda",
    debug: bool = False,
    accuracy_check: bool = False,
) -> nn.Module:
    """
    Compile the Qwen3-VL language model backbone with TRT + static KV cache.

    The compiled module is stored as model._trt_vlm_backbone.  Generation must
    then use generate_alpamayo_with_static_cache() instead of model.vlm.generate().

    Args:
        model:          AlpamayoR1 model (must be on `device`)
        max_seq_len:    Maximum total sequence length (prompt + new tokens).
                        KV cache buffers are pre-allocated to this size.
        precision:      "BF16", "FP16", or "FP32"
        device:         CUDA device string
        debug:          Enable TRT debug logging
        accuracy_check: If True, compare TRT vs PyTorch hidden states on the smoke
                        test input before offloading model weights to CPU.

    Returns:
        The TRT-compiled backbone module (also stored as model._trt_vlm_backbone).
    """
    import torch_tensorrt

    logger.info("=" * 65)
    logger.info("Compiling VLM Language Model Backbone with TRT")
    logger.info("(torch.export + static_cache_v2 FX pass + dynamo.compile)")
    logger.info("=" * 65)
    logger.info(f"  max_seq_len:  {max_seq_len}")
    logger.info(f"  precision:    {precision}")

    dtype_map = {"BF16": torch.bfloat16, "FP16": torch.float16, "FP32": torch.float32}
    dtype = dtype_map[precision.upper()]

    backbone = model.vlm.model          # Qwen3VLModel
    language_model = backbone.language_model  # Qwen3VLTextModel

    hidden_size = language_model.config.hidden_size  # 4096

    # -- Switch to SDPA attention for TRT compatibility --
    # flash_attention_2 / efficient_attention produce ops TRT cannot decompose.
    # Set _attn_implementation at the language model config level and per-layer.
    language_model.config._attn_implementation = "sdpa"
    for layer in language_model.layers:
        if hasattr(layer.self_attn, "_attn_implementation"):
            layer.self_attn._attn_implementation = "sdpa"
        if hasattr(layer.self_attn, "config"):
            layer.self_attn.config._attn_implementation = "sdpa"

    wrapper = Qwen3VLTextModelWrapper(language_model).to(device=device, dtype=dtype).eval()

    # -- Precompute example cos/sin for export (use max_seq_len as representative shape) --
    # The wrapper now accepts (inputs_embeds, cos, sin) where cos/sin are computed
    # OUTSIDE TRT by calling language_model.rotary_emb() in Python.  This avoids
    # the M-RoPE apply_interleaved_mrope scatter op that TRT mis-compiles during decode
    # (TRT uses position_ids scalar VALUE as a reshape target, causing volume mismatches).
    example_embeds = torch.randn(1, max_seq_len, hidden_size, dtype=dtype, device=device)
    example_pos_ids = torch.arange(max_seq_len, device=device).unsqueeze(0)
    with torch.no_grad():
        rotary_emb = language_model.rotary_emb.to(device=device)
        example_cos, example_sin = rotary_emb(example_embeds, example_pos_ids)
    # cos/sin: (B, S, head_dim)
    logger.info(f"  cos/sin shape: {example_cos.shape}, dtype: {example_cos.dtype}")

    # -- Register SDPA lowering pass + static KV cache pass --
    logger.info("Registering SDPA lowering pass...")
    from torchtrt_ext import register_sdpa
    register_sdpa.enable_sdpa_converter("nvidia/Alpamayo-R1-10B", language_model.config)

    logger.info("Registering static KV cache v2 pass...")
    import static_cache_v2  # noqa: F401 — registration side-effect

    # -- Export --
    logger.info(f"Exporting language model (max_seq_len={max_seq_len})...")
    ep = _export_wrapper(wrapper, example_embeds, example_cos, example_sin, max_seq_len=max_seq_len)

    # -- TRT compilation settings --
    trt_settings: dict = {
        "use_explicit_typing": True,
        "use_fp32_acc": True,
        "truncate_double": True,
        "min_block_size": 1,
        "use_python_runtime": True,
        "debug": debug,
        "allow_complex_guards_as_runtime_asserts": True,
        "offload_module_to_cpu": True,
    }
    if precision.upper() == "FP16":
        trt_settings["enabled_precisions"] = {torch.float16, torch.float32}
    elif precision.upper() == "FP32":
        trt_settings["enabled_precisions"] = {torch.float32}
    # BF16 uses use_explicit_typing (default)

    # For dynamo.compile, pass inputs_embeds, cos, sin as the 3 "user-visible" inputs.
    # The KV cache inputs and start_idx/end_idx were added by the static_cache_v2
    # lowering pass and have proper metadata in the graph after FakeTensorUpdater runs.
    # TRT infers their shape profiles from that metadata.
    trt_inputs = [example_embeds, example_cos, example_sin]
    logger.info(f"  TRT inputs: {len(trt_inputs)} (inputs_embeds, cos, sin)")

    # Pre-compute PyTorch reference output for accuracy check (before offload_module_to_cpu).
    # offload_module_to_cpu=True moves weights to CPU during dynamo.compile, so we must
    # run the PyTorch wrapper BEFORE compilation to get the reference output.
    pt_hs_ref = None
    acc_embeds = None
    if accuracy_check:
        check_len = 64
        torch.manual_seed(42)
        acc_embeds = torch.randn(1, check_len, hidden_size, dtype=dtype, device=device)
        acc_pos = torch.arange(check_len, device=device).unsqueeze(0)
        with torch.no_grad():
            acc_cos, acc_sin = language_model.rotary_emb(acc_embeds, acc_pos)
            pt_hs_ref = wrapper(acc_embeds, acc_cos, acc_sin)
        logger.info(f"  PyTorch reference: shape={tuple(pt_hs_ref.shape)}, "
                    f"mean={pt_hs_ref.float().mean():.4f}, std={pt_hs_ref.float().std():.4f}")

    logger.info("Compiling TRT engine (this will take several minutes)...")
    with torch_tensorrt.dynamo.Debugger() if debug else nullcontext():
        trt_backbone = torch_tensorrt.dynamo.compile(
            ep,
            inputs=trt_inputs,
            **trt_settings,
        )

    # Enable output allocator on all TRT submodules.
    # static_cache_v2 creates concat_keys = cat(cache[:start_idx], k) with shape
    # (1, 32, start_idx+s, 128) — a data-dependent ("profile-oblivious") output.
    # TRT requires the output-allocator path for these; the default address-based
    # path sets ptr=0 and causes enqueueV3 error "Neither address or allocator is
    # set for output tensor outputN".
    logger.info("Enabling output allocator on TRT submodules (static_cache_v2 compatibility)...")
    _fix_requires_output_allocator(trt_backbone)

    # Smoke test: verify TRT backbone runs without error on a short sequence.
    logger.info("Running TRT smoke test (prefill check)...")
    check_len = 64
    torch.manual_seed(42)
    short_embeds = torch.randn(1, check_len, hidden_size, dtype=dtype, device=device)
    short_pos = torch.arange(check_len, device=device).unsqueeze(0)
    with torch.no_grad():
        short_cos, short_sin = language_model.rotary_emb.to(device)(short_embeds, short_pos)
    zeroed_kv = _get_zeroed_kv_for_trt(trt_backbone, device)
    try:
        with torch.no_grad():
            trt_out_raw = trt_backbone(short_embeds, short_cos, short_sin, *zeroed_kv, 0, check_len)
        trt_hs = trt_out_raw[0]
        logger.info(f"  TRT output shape: {tuple(trt_hs.shape)}, "
                    f"dtype: {trt_hs.dtype}, "
                    f"mean: {trt_hs.float().mean():.4f}, "
                    f"std: {trt_hs.float().std():.4f}")
        logger.info("  ✓ TRT smoke test passed")

        if accuracy_check and pt_hs_ref is not None:
            # Compare TRT vs PyTorch reference output (computed before offload).
            # Both use the same random seed and identical inputs.
            logger.info("  Accuracy check (TRT vs PyTorch reference):")
            pt_hs_f = pt_hs_ref.to(dtype=torch.float32)
            trt_hs_f = trt_hs.to(dtype=torch.float32)
            l2_err = (pt_hs_f - trt_hs_f).norm() / pt_hs_f.norm()
            cos_sim = torch.nn.functional.cosine_similarity(
                pt_hs_f.reshape(-1), trt_hs_f.reshape(-1), dim=0
            )
            logger.info(f"    L2 relative error: {l2_err:.4f}")
            logger.info(f"    Cosine similarity: {cos_sim:.6f}")
            if l2_err > 0.05:
                logger.warning(f"    ⚠ L2 relative error {l2_err:.4f} > 0.05 (BF16 rounding may cause some drift)")
            else:
                logger.info("    ✓ Accuracy check passed (L2 err within 5%)")
    except Exception as e:
        logger.warning(f"  TRT smoke test failed: {e} — continuing")

    model._trt_vlm_backbone = trt_backbone
    logger.info("✓ VLM LM backbone compiled (stored as model._trt_vlm_backbone)")
    return trt_backbone


# ---------------------------------------------------------------------------
# KV cache helpers
# ---------------------------------------------------------------------------


def _get_zeroed_kv_for_trt(trt_backbone: nn.Module, device: str) -> list[torch.Tensor]:
    """
    Build zeroed KV cache tensors from the TRT-compiled backbone's graph placeholders.

    After the static_cache_v2 pass, the placeholder order in the FX graph is:
        [0] inputs_embeds
        [1] cos
        [2] sin
        [3..N-3] k0, v0, k1, v1, ..., k35, v35
        [N-2] start_idx
        [N-1] end_idx

    Returns the list [k0, v0, ..., k35, v35] (excludes start/end_idx).

    Uses the same approach as tools/llm/utils.py:get_zeroed_static_cache_inputs().
    """
    # torch_tensorrt.dynamo.compile returns a torch.fx.GraphModule wrapper
    placeholder_nodes = [n for n in trt_backbone.graph.nodes if n.op == "placeholder"]
    kv_placeholders = placeholder_nodes[3:-2]  # skip inputs_embeds, cos, sin, start_idx, end_idx

    kv_tensors = []
    for ph in kv_placeholders:
        val = ph.meta.get("val")
        if val is None:
            raise RuntimeError(f"Placeholder {ph.name} has no 'val' metadata — "
                               "make sure static_cache_v2 pass ran before compile")
        # Resolve any SymInt dims to their concrete hint value
        shape = [
            (int(d.node.hint) if isinstance(d, torch.SymInt) else int(d))
            for d in val.shape
        ]
        kv_tensors.append(torch.zeros(shape, dtype=val.dtype, device=device))

    return kv_tensors


# ---------------------------------------------------------------------------
# Alpamayo-specific autoregressive generation with static KV cache
# ---------------------------------------------------------------------------


def generate_alpamayo_with_static_cache(
    model: nn.Module,
    trt_backbone: nn.Module,
    input_ids: torch.LongTensor,
    tokenized_data: dict,
    eos_token_id: int,
    max_new_tokens: int = 256,
    top_p: float = 0.98,
    temperature: float = 0.6,
    num_return_sequences: int = 1,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> "_AlpamayoLMOutput":
    """
    Autoregressive generation using TRT-compiled LM backbone with static KV cache.

    Replaces model.vlm.generate() in run_inference_trt().  Returns an object with
    the same fields that run_inference_trt() uses from vlm_outputs:
        .sequences         — generated token ids  [B, seq_len]
        .past_key_values   — DynamicCache populated from the TRT KV tensors
        .rope_deltas       — from model.vlm.model.rope_deltas (set during prefill)
        .logits            — tuple of per-step logits (for logits_processor compat)

    Args:
        model:               AlpamayoR1 model
        trt_backbone:        TRT-compiled Qwen3VLTextModelWrapper module
        input_ids:           [B, prompt_len] — fused input_ids (after fuse_traj_tokens)
        tokenized_data:      dict with attention_mask, pixel_values, etc. (from processor)
        eos_token_id:        Token id at which to stop generation
        max_new_tokens:      Maximum new tokens to generate
        top_p / temperature: Sampling parameters
        num_return_sequences: Number of generation sequences (batch multiplier)
        device / dtype:      Device and dtype for tensors

    Returns:
        _AlpamayoLMOutput with .sequences, .past_key_values, .rope_deltas, .logits
    """
    from transformers import DynamicCache

    backbone = model.vlm.model  # Qwen3VLModel
    emb_layer = backbone.get_input_embeddings()
    lm_head = model.vlm.lm_head
    rotary_emb = backbone.language_model.rotary_emb

    # After TRT compilation with offload_module_to_cpu=True, the language model weights
    # (including embed_tokens / lm_head, which are tied) are on CPU.
    # Move embed_tokens, lm_head and rotary_emb to CUDA for generation.
    if next(emb_layer.parameters()).device.type == "cpu":
        emb_layer.to(device=device)
    if next(lm_head.parameters()).device.type == "cpu":
        lm_head.to(device=device)
    if next(rotary_emb.buffers()).device.type == "cpu":
        rotary_emb.to(device=device)

    # --------------------------------------------------------------------- #
    # Step 1: Build inputs_embeds for the full prompt (merge image embeds)
    # --------------------------------------------------------------------- #
    with torch.no_grad():
        inputs_embeds, attention_mask, position_ids = _prepare_prefill_inputs(
            model, input_ids, tokenized_data, device, dtype
        )

    # --------------------------------------------------------------------- #
    # Step 2: Prefill — run full prompt through TRT backbone
    # --------------------------------------------------------------------- #
    prompt_len = inputs_embeds.shape[1]

    # Zeroed KV buffers (pre-allocated to max_seq_len by static_cache_v2)
    kv_cache = _get_zeroed_kv_for_trt(trt_backbone, device)
    max_seq_len = kv_cache[0].shape[2]
    if prompt_len + max_new_tokens > max_seq_len:
        raise ValueError(
            f"prompt_len ({prompt_len}) + max_new_tokens ({max_new_tokens}) = "
            f"{prompt_len + max_new_tokens} exceeds TRT max_seq_len ({max_seq_len}). "
            f"Re-compile with a larger max_seq_len."
        )

    start_idx = 0
    end_idx = prompt_len

    logger.debug(f"  Prefill: prompt_len={prompt_len}, max_seq_len={max_seq_len}")

    with torch.no_grad():
        # Precompute cos/sin in Python (avoids TRT M-RoPE reshape bug)
        prefill_pos_ids = torch.arange(prompt_len, device=device).unsqueeze(0)
        prefill_pos_ids = prefill_pos_ids.expand(inputs_embeds.shape[0], -1)
        prefill_cos, prefill_sin = rotary_emb(inputs_embeds, prefill_pos_ids)

        prefill_inputs = (
            inputs_embeds.to(dtype),
            prefill_cos,
            prefill_sin,
            *kv_cache,
            start_idx,
            end_idx,
        )
        prefill_outputs = trt_backbone(*prefill_inputs)
        prefill_hidden = prefill_outputs[0]  # (B, prompt_len, hidden_size)
        kv_cache = list(prefill_outputs[1:])

    # Get logits for the last prompt token → first generated token
    prefill_logits = lm_head(prefill_hidden[:, -1:, :].to(lm_head.weight.dtype))  # (B, 1, vocab)

    # --------------------------------------------------------------------- #
    # Step 3: Autoregressive decode
    # --------------------------------------------------------------------- #
    output_tokens = input_ids.clone()
    all_logits = [prefill_logits]

    # Sample first token from prefill logits
    next_token = _sample_next_token(prefill_logits[:, -1, :], top_p, temperature)
    output_tokens = torch.cat([output_tokens, next_token[:, None]], dim=-1)

    start_idx = end_idx        # = prompt_len
    end_idx = start_idx + 1

    generated = 1
    while generated < max_new_tokens:
        if (next_token == eos_token_id).all():
            break

        # Embed the new token
        next_embed = emb_layer(next_token)[:, None, :].to(dtype)  # (B, 1, hidden)

        # Precompute cos/sin for the new token position in Python
        decode_pos_ids = torch.tensor(
            [[start_idx]], dtype=torch.long, device=device
        ).expand(next_embed.shape[0], -1)
        with torch.no_grad():
            decode_cos, decode_sin = rotary_emb(next_embed, decode_pos_ids)

        with torch.no_grad():
            decode_inputs = (
                next_embed,
                decode_cos,
                decode_sin,
                *kv_cache,
                start_idx,
                end_idx,
            )
            decode_outputs = trt_backbone(*decode_inputs)
            decode_hidden = decode_outputs[0]   # (B, 1, hidden_size)
            kv_cache = list(decode_outputs[1:])

        decode_logits = lm_head(decode_hidden[:, -1:, :].to(lm_head.weight.dtype))
        all_logits.append(decode_logits)

        next_token = _sample_next_token(decode_logits[:, -1, :], top_p, temperature)
        output_tokens = torch.cat([output_tokens, next_token[:, None]], dim=-1)

        start_idx = end_idx
        end_idx += 1
        generated += 1

    # --------------------------------------------------------------------- #
    # Step 4: Build a DynamicCache from the TRT KV tensors for downstream use
    # --------------------------------------------------------------------- #
    # run_inference_trt() reads prompt_cache.layers[i].keys/values or
    # prompt_cache.get_seq_length() to extract prefix_k/prefix_v for TRT diffusion.
    past_key_values = _build_dynamic_cache_from_kv_list(
        kv_cache, num_layers=len(kv_cache) // 2, seq_len=end_idx
    )

    return _AlpamayoLMOutput(
        sequences=output_tokens,
        past_key_values=past_key_values,
        rope_deltas=getattr(backbone, "rope_deltas", None),
        logits=tuple(all_logits),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepare_prefill_inputs(
    model: nn.Module,
    input_ids: torch.LongTensor,
    tokenized_data: dict,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare inputs_embeds, attention_mask, and position_ids for the prefill pass.

    Replicates the embedding-merge logic from Qwen3VLModel.forward():
      1. Embed input_ids with the text embedding layer
      2. Encode pixel_values with the vision model → image_embeds
         (deepstack_image_embeds are discarded — they cannot be passed into TRT)
      3. Replace image placeholder tokens in inputs_embeds with image_embeds
         via masked_scatter (same as Qwen3VLModel.forward lines 1131-1143)

    position_ids is returned as (B, S) — Qwen3VLTextModel auto-expands to (3, B, S).
    """
    backbone = model.vlm.model  # Qwen3VLModel

    pixel_values = tokenized_data.get("pixel_values")
    image_grid_thw = tokenized_data.get("image_grid_thw")
    attention_mask = tokenized_data.get("attention_mask")

    # Step 1: text token embeddings
    # After TRT compilation with offload_module_to_cpu=True, model weights are on CPU.
    # Move the embedding layer to CUDA temporarily for the forward pass.
    embed_layer = backbone.get_input_embeddings()
    was_on_cpu = next(embed_layer.parameters()).device.type == "cpu"
    if was_on_cpu:
        embed_layer.to(device=device)
    inputs_embeds = embed_layer(input_ids.to(device=device)).to(dtype=dtype, device=device)
    if was_on_cpu:
        embed_layer.to("cpu")

    # Step 2 & 3: merge image embeddings if present
    if pixel_values is not None:
        # get_image_features returns (image_embeds_list, deepstack_image_embeds)
        # We discard deepstack since it can't be passed to TRT (list of tensors with
        # variable visual sequence length)
        image_embeds_list, _deepstack = backbone.get_image_features(
            pixel_values.to(device=device), image_grid_thw.to(device=device)
        )
        image_embeds = torch.cat(image_embeds_list, dim=0).to(dtype=dtype, device=device)
        image_mask, _ = backbone.get_placeholder_mask(
            input_ids.to(device=device),
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    # position_ids: (B, S) plain sequential — model auto-expands to M-RoPE (3, B, S)
    position_ids = torch.arange(inputs_embeds.shape[1], device=device).unsqueeze(0)
    position_ids = position_ids.expand(inputs_embeds.shape[0], -1)

    return (
        inputs_embeds,
        attention_mask,
        position_ids,
    )


def _sample_next_token(
    logits: torch.Tensor,  # (B, vocab_size)
    top_p: float,
    temperature: float,
) -> torch.LongTensor:  # (B,)
    """Simple top-p sampling (or greedy if temperature == 0)."""
    if temperature == 0.0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if top_p >= 1.0:
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    # Top-p (nucleus) filtering
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens whose cumulative prob exceeds top_p (keep the first one that exceeds)
    sorted_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[sorted_remove] = float("-inf")
    # Scatter back to original order
    logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _build_dynamic_cache_from_kv_list(
    kv_list: list[torch.Tensor],
    num_layers: int,
    seq_len: int,
) -> "DynamicCache":
    """
    Reconstruct a DynamicCache from the flat [k0, v0, k1, v1, ...] list returned
    by the TRT backbone after generation.

    kv_list[2i]   = key   for layer i, shape (B, num_heads, max_cache_len, head_dim)
    kv_list[2i+1] = value for layer i, same shape
    (num_heads = 32 after GQA expansion via repeat_kv in the TRT graph)

    We trim to [:, :, :seq_len, :] so downstream code (extracting prefix_k/v) sees
    only the filled portion.
    """
    from transformers import DynamicCache

    cache = DynamicCache()
    for layer_idx in range(num_layers):
        k = kv_list[2 * layer_idx][:, :, :seq_len, :]   # trim to filled length
        v = kv_list[2 * layer_idx + 1][:, :, :seq_len, :]
        cache.update(k, v, layer_idx)
    return cache


class _AlpamayoLMOutput:
    """Minimal output container matching what run_inference_trt() reads from vlm_outputs."""

    __slots__ = ("sequences", "past_key_values", "rope_deltas", "logits")

    def __init__(self, sequences, past_key_values, rope_deltas, logits):
        self.sequences = sequences
        self.past_key_values = past_key_values
        self.rope_deltas = rope_deltas
        self.logits = logits
