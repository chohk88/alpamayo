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
TRT Engine Test: save raw .trt engines to disk, reload without torch_tensorrt.

Validates the full serialize → save → load → infer pipeline for both the vision
encoder and the diffusion step using only ``tensorrt`` + ``torch`` at inference
time (no torch_tensorrt required after saving).

Two modes:

  --save   Compile and serialize engines to .trt files using torch_tensorrt.
  --infer  Load .trt files with pure TensorRT and run inference vs PyTorch baseline.
  (default) Run both save and infer in sequence.

Usage:
    # Save engines (requires torch_tensorrt):
    PYTHONPATH=src python -m alpamayo_r1.test_trt_engines \\
        --save --engine-dir /tmp/alpamayo_engines

    # Reload and infer (only requires tensorrt):
    PYTHONPATH=src python -m alpamayo_r1.test_trt_engines \\
        --infer --engine-dir /tmp/alpamayo_engines

    # Both in one go (default):
    PYTHONPATH=src python -m alpamayo_r1.test_trt_engines \\
        --engine-dir /tmp/alpamayo_engines
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import time
from typing import Any

import einops
import numpy as np
import torch
import torch.nn as nn
from transformers import StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Wrappers: TRTEngineRunner → call signatures used by run_inference_trt
# =============================================================================


class VisionTRTEngineWrapper:
    """
    Wraps TRTEngineRunner to match the VisualFixedGrid forward signature:
        (hidden_states, grid_thw=None) -> (main_features, deepstack_feature_list)

    The saved vision engine was compiled from VisualFixedGrid which returns:
        output[0] = hidden_states  [total_patches, hidden_size]
        output[1..N] = deepstack features (one per deepstack layer)
    """

    def __init__(self, runner):
        self.runner = runner

    def forward(self, hidden_states: torch.Tensor, grid_thw=None):
        outputs = self.runner(hidden_states)
        # First output is main features, rest are deepstack features
        main_features = outputs[0]
        deepstack_features = outputs[1:]
        return main_features, deepstack_features

    def __call__(self, hidden_states: torch.Tensor, grid_thw=None):
        return self.forward(hidden_states, grid_thw)


class DiffusionTRTEngineWrapper:
    """
    Wraps TRTEngineRunner to match the StaticKVDiffusionStepModule forward signature:
        (x, t, prefix_k, prefix_v, position_ids, attention_mask) -> action_tensor

    The saved diffusion engine was compiled from StaticKVDiffusionStepModule which
    returns a single output: the predicted action [B, *action_space_dims].
    """

    def __init__(self, runner):
        self.runner = runner

    def __call__(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        result = self.runner(x, t, prefix_k, prefix_v, position_ids, attention_mask)
        # TRTEngineRunner returns list[Tensor]; torch_tensorrt.load module returns Tensor
        if isinstance(result, (list, tuple)):
            return result[0]
        return result


# =============================================================================
# Data / Model Loading  (identical to test_combined_trt.py)
# =============================================================================


def load_test_data(clip_id: str, t0_us: int = 5_100_000) -> tuple[dict, list]:
    from alpamayo_r1 import helper
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    logger.info(f"Loading dataset for clip_id: {clip_id}...")
    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    return data, messages


def prepare_model_inputs(model, data: dict, messages: list, device: str = "cuda") -> callable:
    from alpamayo_r1 import helper

    processor = helper.get_processor(model.tokenizer)

    def create_inputs():
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"].clone(),
            "ego_history_rot": data["ego_history_rot"].clone(),
        }
        return helper.to_device(model_inputs, device)

    return create_inputs


def compute_trajectory_metrics(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor) -> dict[str, float]:
    gt_xy = gt_xyz.cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    ade_per_sample = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    fde_per_sample = np.linalg.norm(pred_xy[:, :, -1] - gt_xy[:, -1], axis=1)
    return {
        "min_ade": float(ade_per_sample.min()),
        "mean_ade": float(ade_per_sample.mean()),
        "min_fde": float(fde_per_sample.min()),
        "mean_fde": float(fde_per_sample.mean()),
    }


# =============================================================================
# Dry-run: measure prefix_seq_len
# =============================================================================


def measure_prefix_seq_len(model, create_inputs_fn: callable, seed: int = 42) -> int:
    """Run one VLM generation pass and return the resulting KV-cache length."""
    from alpamayo_r1.models.alpamayo_r1 import ExpertLogitsProcessor
    from alpamayo_r1.models.token_utils import StopAfterEOS, to_special_token

    torch.cuda.manual_seed_all(seed)
    _inputs = create_inputs_fn()
    _tokenized = _inputs["tokenized_data"]
    _input_ids = _tokenized.pop("input_ids")
    _input_ids = model.fuse_traj_tokens(
        _input_ids,
        {
            "ego_history_xyz": _inputs["ego_history_xyz"],
            "ego_history_rot": _inputs["ego_history_rot"],
        },
    )
    _eos_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))

    _gen_cfg = model.vlm.generation_config
    _gen_cfg.top_p = 0.98
    _gen_cfg.temperature = 0.6
    _gen_cfg.do_sample = True
    _gen_cfg.num_return_sequences = 1
    _gen_cfg.max_new_tokens = 256
    _gen_cfg.output_logits = True
    _gen_cfg.return_dict_in_generate = True
    _gen_cfg.top_k = None
    _gen_cfg.pad_token_id = model.tokenizer.pad_token_id

    with torch.no_grad():
        _vlm_out = model.vlm.generate(
            input_ids=_input_ids,
            generation_config=_gen_cfg,
            stopping_criteria=StoppingCriteriaList([StopAfterEOS(eos_token_id=_eos_id)]),
            logits_processor=LogitsProcessorList(
                [
                    ExpertLogitsProcessor(
                        traj_token_offset=model.config.traj_token_start_idx,
                        traj_vocab_size=model.config.traj_vocab_size,
                    )
                ]
            ),
            **_tokenized,
        )

    seq_len = _vlm_out.past_key_values.get_seq_length()
    del _inputs, _tokenized, _vlm_out
    logger.info(f"  prefix_seq_len = {seq_len}")
    return seq_len


# =============================================================================
# Save engines
# =============================================================================


def save_engines(
    model: nn.Module,
    model_inputs: dict[str, Any],
    engine_dir: str | pathlib.Path,
    max_prefix_len: int,
) -> bool:
    """
    Serialize the vision and diffusion engines to disk using torch_tensorrt.

    Writes:
        <engine_dir>/vision.trt          — vision encoder engine
        <engine_dir>/vision.trt.json     — metadata sidecar
        <engine_dir>/diffusion.trt       — diffusion step engine
        <engine_dir>/diffusion.trt.json  — metadata sidecar

    Args:
        model:           AlpamayoR1 model
        model_inputs:    dict with tokenized_data (pixel_values, image_grid_thw)
        engine_dir:      Directory to write engine files
        max_prefix_len:  Max VLM KV prefix length for diffusion engine
    Returns:
        True if both engines were saved successfully.
    """
    from alpamayo_r1.trt.diffusion import save_diffusion_engine
    from alpamayo_r1.trt.vision import save_vision_engine

    engine_dir = pathlib.Path(engine_dir)
    engine_dir.mkdir(parents=True, exist_ok=True)

    vision_path = engine_dir / "vision.trt"
    diffusion_path = engine_dir / "diffusion.trt"

    # --- Vision ---
    logger.info("\n" + "=" * 60)
    logger.info("Saving Vision Encoder engine")
    logger.info("=" * 60)
    ok_vision = save_vision_engine(
        model.vlm.model.visual,
        model_inputs,
        path=str(vision_path),
        device="cuda",
    )
    if not ok_vision:
        logger.error("Failed to save vision engine")
        return False

    # --- Diffusion ---
    logger.info("\n" + "=" * 60)
    logger.info("Saving Diffusion Step engine")
    logger.info("=" * 60)
    ok_diffusion = save_diffusion_engine(
        model,
        path=str(diffusion_path),
        max_prefix_len=max_prefix_len,
        min_prefix_len=1,
        device="cuda",
    )
    if not ok_diffusion:
        logger.error("Failed to save diffusion engine")
        return False

    logger.info("\n✓ Both engines saved:")
    logger.info(f"  {vision_path}")
    logger.info(f"  {diffusion_path}")
    return True


# =============================================================================
# Load engines (pure TRT)
# =============================================================================


def load_vision_engine(
    engine_dir: str | pathlib.Path,
    device: str = "cuda",
) -> VisionTRTEngineWrapper:
    """
    Load only the vision engine from disk using pure TensorRT.

    Args:
        engine_dir: Directory containing vision.trt
        device:     CUDA device string

    Returns:
        VisionTRTEngineWrapper
    """
    from alpamayo_r1.trt.engine_io import TRTEngineRunner

    engine_dir = pathlib.Path(engine_dir)
    vision_path = engine_dir / "vision.trt"

    if not vision_path.exists():
        raise FileNotFoundError(f"Vision engine not found: {vision_path}")

    logger.info(f"Loading vision engine from {vision_path} (pure TRT)")
    vision_runner = TRTEngineRunner(vision_path, device=device)
    vision_wrapper = VisionTRTEngineWrapper(vision_runner)
    logger.info("✓ Vision engine loaded")
    return vision_wrapper


def load_engines(
    engine_dir: str | pathlib.Path,
    device: str = "cuda",
) -> tuple[VisionTRTEngineWrapper, DiffusionTRTEngineWrapper]:
    """
    Load vision and diffusion engines from disk.

    Both engines are raw .trt files loaded via TRTEngineRunner — no torch_tensorrt
    required at inference time.

    Args:
        engine_dir: Directory containing vision.trt and diffusion.trt
        device:     CUDA device string

    Returns:
        (vision_wrapper, diffusion_wrapper)
    """
    from alpamayo_r1.trt.engine_io import TRTEngineRunner

    engine_dir = pathlib.Path(engine_dir)
    vision_path    = engine_dir / "vision.trt"
    diffusion_path = engine_dir / "diffusion.trt"

    if not vision_path.exists():
        raise FileNotFoundError(f"Vision engine not found: {vision_path}")
    if not diffusion_path.exists():
        raise FileNotFoundError(f"Diffusion engine not found: {diffusion_path}")

    logger.info(f"Loading vision engine from {vision_path} (pure TRT)")
    vision_runner = TRTEngineRunner(vision_path, device=device)
    vision_wrapper = VisionTRTEngineWrapper(vision_runner)

    logger.info(f"Loading diffusion engine from {diffusion_path} (pure TRT)")
    diffusion_runner = TRTEngineRunner(diffusion_path, device=device)
    diffusion_wrapper = DiffusionTRTEngineWrapper(diffusion_runner)

    logger.info("✓ Both engines loaded")
    return vision_wrapper, diffusion_wrapper


# =============================================================================
# Inference (same logic as test_combined_trt.run_inference_trt)
# =============================================================================


def run_inference_pure_trt(
    model,
    create_inputs_fn: callable,
    trt_vision,
    trt_diffusion,
    seed: int = 42,
    num_traj_samples: int = 1,
    max_generation_length: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, dict, float]:
    """
    Run inference with pure-TRT vision and diffusion wrappers.

    The VLM generation still uses PyTorch (autoregressive generation with KV
    cache cannot be compiled to a single static TRT engine).
    """
    from alpamayo_r1.models.alpamayo_r1 import ExpertLogitsProcessor
    from alpamayo_r1.models.token_utils import (
        StopAfterEOS,
        extract_text_tokens,
        replace_padding_after_eos,
        to_special_token,
    )

    torch.cuda.manual_seed_all(seed)
    model_inputs = create_inputs_fn()

    dtype = torch.bfloat16
    device = "cuda"

    start_time = time.perf_counter()

    with torch.autocast("cuda", dtype=dtype):
        ego_history_xyz = model_inputs["ego_history_xyz"]
        ego_history_rot = model_inputs["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        tokenized_data = model_inputs["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")

        traj_data_vlm = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = model.fuse_traj_tokens(input_ids, traj_data_vlm)

        # --- Optionally replace vision forward with pure-TRT wrapper ---
        original_vision_forward = None
        if trt_vision is not None:
            original_vision_forward = model.vlm.model.visual.forward
            model.vlm.model.visual.forward = trt_vision.forward

        # --- Autoregressive VLM generation (PyTorch with KV cache) ---
        eos_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
        generation_config = model.vlm.generation_config
        generation_config.top_p = 0.98
        generation_config.temperature = 0.6
        generation_config.do_sample = True
        generation_config.num_return_sequences = num_traj_samples
        generation_config.max_new_tokens = max_generation_length
        generation_config.output_logits = True
        generation_config.return_dict_in_generate = True
        generation_config.top_k = None
        generation_config.pad_token_id = model.tokenizer.pad_token_id

        stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
        logits_processor = LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=model.config.traj_token_start_idx,
                    traj_vocab_size=model.config.traj_vocab_size,
                )
            ]
        )

        vlm_outputs = model.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **tokenized_data,
        )
        vlm_outputs.rope_deltas = model.vlm.model.rope_deltas

        if original_vision_forward is not None:
            model.vlm.model.visual.forward = original_vision_forward

        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=model.tokenizer.pad_token_id,
        )

        # --- Compute prefix boundary ---
        b_star = vlm_outputs.sequences.shape[0]
        traj_future_start_mask = vlm_outputs.sequences == eos_token_id
        has_traj_future_start = traj_future_start_mask.any(dim=1)
        traj_future_start_positions = traj_future_start_mask.int().argmax(dim=1)
        last_token_positions = torch.full(
            (b_star,), vlm_outputs.sequences.shape[1] - 1, device=device
        )
        valid_token_pos_id = torch.where(
            has_traj_future_start, traj_future_start_positions, last_token_positions
        )
        offset = valid_token_pos_id + 1

        n_diffusion_tokens = model.action_space.get_action_space_dims()[0]

        # --- Build position_ids and attention_mask ---
        prompt_cache = vlm_outputs.past_key_values
        prefill_seq_len = prompt_cache.get_seq_length()

        position_ids = torch.arange(n_diffusion_tokens, device=device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star).clone()
        delta = vlm_outputs.rope_deltas + offset[:, None]
        position_ids = position_ids + delta.to(device)

        NEG_INF = torch.finfo(torch.float32).min
        attention_mask = torch.zeros(
            b_star, 1, n_diffusion_tokens,
            prefill_seq_len + n_diffusion_tokens,
            dtype=torch.float32, device=device,
        )
        for i in range(b_star):
            attention_mask[i, :, :, offset[i]: -n_diffusion_tokens] = NEG_INF
        attention_mask = attention_mask.to(dtype)

        # --- Extract stacked KV tensors ---
        prefix_k = torch.stack(
            [layer.keys for layer in prompt_cache.layers], dim=0
        )  # [L, B, H, S, D]
        prefix_v = torch.stack([layer.values for layer in prompt_cache.layers], dim=0)
        logger.info(f"  prefix_k shape: {prefix_k.shape}")

        # --- Diffusion sampling ---
        forward_kwargs = {}
        if model.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        if trt_diffusion is not None:
            def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                return trt_diffusion(
                    x.to(dtype), t.to(dtype), prefix_k, prefix_v, position_ids, attention_mask
                )
        else:
            def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                b = x.shape[0]
                future_token_embeds = model.action_in_proj(x, t)
                if future_token_embeds.dim() == 2:
                    future_token_embeds = future_token_embeds.view(b, n_diffusion_tokens, -1)
                expert_out = model.expert(
                    inputs_embeds=future_token_embeds,
                    position_ids=position_ids,
                    past_key_values=prompt_cache,
                    attention_mask=attention_mask,
                    use_cache=True,
                    **forward_kwargs,
                )
                prompt_cache.crop(prefill_seq_len)
                last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
                return model.action_out_proj(last_hidden).view(
                    -1, *model.action_space.get_action_space_dims()
                )

        total_batch = B * num_traj_samples
        sampled_action = model.diffusion.sample(
            batch_size=total_batch,
            step_fn=step_fn,
            device=device,
            return_all_steps=False,
        )

        # --- Convert to trajectories ---
        hist_xyz_rep = einops.repeat(
            ego_history_xyz[:, -1], "b ... -> (b n) ...", n=num_traj_samples
        )
        hist_rot_rep = einops.repeat(
            ego_history_rot[:, -1], "b ... -> (b n) ...", n=num_traj_samples
        )
        pred_xyz, pred_rot = model.action_space.action_to_traj(
            sampled_action, hist_xyz_rep, hist_rot_rep
        )
        pred_xyz = einops.rearrange(
            pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=1, nj=num_traj_samples
        )
        pred_rot = einops.rearrange(
            pred_rot, "(b ns nj) ... -> b ns nj ...", ns=1, nj=num_traj_samples
        )

        extra = extract_text_tokens(model.tokenizer, vlm_outputs.sequences)
        for k in extra:
            extra[k] = np.array(extra[k]).reshape([input_ids.shape[0], 1, num_traj_samples])

    inference_time = time.perf_counter() - start_time
    return pred_xyz, pred_rot, extra, inference_time


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pure-TRT inference test: save engines, reload without torch_tensorrt"
    )
    parser.add_argument("--model_path", type=str, default="nvidia/Alpamayo-R1-10B")
    parser.add_argument(
        "--clip_id", type=str, default="030c760c-ae38-49aa-9ad8-f5650a545d26"
    )
    parser.add_argument(
        "--engine-dir", type=str, default="/tmp/alpamayo_engines",
        help="Directory to save/load .trt engine files"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save", action="store_true",
        help="Serialize engines to disk (requires torch_tensorrt)"
    )
    parser.add_argument(
        "--infer", action="store_true",
        help="Load engines and run inference (requires only tensorrt + torch)"
    )
    parser.add_argument(
        "--vision-only", action="store_true",
        help="Only test vision engine (skip diffusion). Diffusion uses PyTorch KV-cache path."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Default: run both save and infer
    do_save  = args.save  or (not args.save and not args.infer)
    do_infer = args.infer or (not args.save and not args.infer)

    logger.info("=" * 70)
    logger.info("PURE-TRT INFERENCE TEST")
    logger.info(f"  engine_dir:   {args.engine_dir}")
    logger.info("  precision:    BF16")
    logger.info(f"  do_save:      {do_save}")
    logger.info(f"  do_infer:     {do_infer}")
    logger.info(f"  vision_only:  {args.vision_only}")
    logger.info("=" * 70)

    # Always need the model (to run PyTorch VLM generation and baseline)
    logger.info(f"\nLoading model: {args.model_path}...")
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

    model = AlpamayoR1.from_pretrained(args.model_path, dtype=torch.bfloat16).to("cuda")
    model.eval()

    data, messages = load_test_data(args.clip_id)
    gt_xyz = data["ego_future_xyz"]
    create_inputs_fn = prepare_model_inputs(model, data, messages, device="cuda")
    model_inputs = create_inputs_fn()

    # -------------------------------------------------------------------------
    # Determine prefix length (needed for save; skipped for infer-only)
    # -------------------------------------------------------------------------
    max_prefix_len = None
    if do_save:
        logger.info("\nMeasuring prefix sequence length...")
        max_prefix_len = measure_prefix_seq_len(model, create_inputs_fn, seed=args.seed)
        logger.info(f"  max_prefix_len = {max_prefix_len}")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    if do_save:
        ok = save_engines(
            model,
            model_inputs,
            engine_dir=args.engine_dir,
            max_prefix_len=max_prefix_len,
        )
        if not ok:
            return 1

    # -------------------------------------------------------------------------
    # PyTorch baseline (no-cache path — same algorithm as TRT path)
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("PYTORCH NO-CACHE BASELINE")
    logger.info("=" * 70)

    pred_xyz_pt, _, _, pt_time = run_inference_pure_trt(
        model, create_inputs_fn,
        trt_vision=None, trt_diffusion=None,
        seed=args.seed,
    )
    pt_metrics = compute_trajectory_metrics(pred_xyz_pt, gt_xyz)
    logger.info(f"  Inference time: {pt_time * 1000:.2f} ms")
    logger.info(f"  minADE: {pt_metrics['min_ade']:.4f} m")

    if not do_infer:
        logger.info("\n--save only: skipping pure-TRT inference")
        return 0

    # -------------------------------------------------------------------------
    # Load engines (pure TRT) and run inference
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("LOADING ENGINES (pure TRT — no torch_tensorrt)")
    logger.info("=" * 70)

    if args.vision_only:
        trt_vision = load_vision_engine(args.engine_dir, device="cuda")
        trt_diffusion = None
        logger.info("Vision-only mode: diffusion will use PyTorch KV-cache path")
        infer_label = "PURE-TRT VISION + PYTORCH DIFFUSION (KV-cache)"
    else:
        trt_vision, trt_diffusion = load_engines(args.engine_dir, device="cuda")
        infer_label = "PURE-TRT INFERENCE (Vision: TRT, Language: PyTorch, Diffusion: TRT)"

    logger.info("\n" + "=" * 70)
    logger.info(infer_label)
    logger.info("=" * 70)

    pred_xyz_trt, _, extra_trt, trt_time = run_inference_pure_trt(
        model, create_inputs_fn,
        trt_vision=trt_vision, trt_diffusion=trt_diffusion,
        seed=args.seed,
    )
    trt_metrics = compute_trajectory_metrics(pred_xyz_trt, gt_xyz)
    logger.info(f"  Inference time: {trt_time * 1000:.2f} ms")
    logger.info(f"  minADE: {trt_metrics['min_ade']:.4f} m")
    logger.info(f"  CoC: {extra_trt['cot'][0][0, 0][:120]}...")

    # -------------------------------------------------------------------------
    # Comparison
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON")
    logger.info("=" * 70)

    diff = torch.abs(pred_xyz_pt.cpu().float() - pred_xyz_trt.cpu().float())
    logger.info(f"  PyTorch no-cache vs Pure-TRT:")
    logger.info(f"    Max trajectory diff:  {diff.max():.6f} m")
    logger.info(f"    Mean trajectory diff: {diff.mean():.6f} m")
    ade_diff = abs(pt_metrics["min_ade"] - trt_metrics["min_ade"])
    logger.info(f"    ADE difference:       {ade_diff:.4f} m")

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  PyTorch no-cache minADE: {pt_metrics['min_ade']:.4f} m  ({pt_time*1000:.0f} ms)")
    logger.info(f"  Pure-TRT minADE:         {trt_metrics['min_ade']:.4f} m  ({trt_time*1000:.0f} ms)")
    logger.info(f"  ADE diff:                {ade_diff:.4f} m")

    mean_diff = diff.mean().item()
    if mean_diff < 0.05:
        logger.info("\n✓ Pure-TRT matches PyTorch no-cache (mean diff < 5cm)")
    else:
        logger.warning(f"\n⚠ Mean trajectory diff = {mean_diff:.4f}m — needs investigation")

    if ade_diff < 0.15:
        logger.info(f"✓ ADE within 15cm of PyTorch baseline ({ade_diff:.4f}m)")
        logger.info("  Pure-TRT inference is production-ready.")
        return 0
    else:
        logger.warning(f"⚠ ADE diff ({ade_diff:.4f}m) exceeds 15cm threshold")
        return 1


if __name__ == "__main__":
    exit(main())
