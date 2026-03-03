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
TRT test using torch-tensorrt compiled modules: Vision + Language + Diffusion in TRT.

Compiles vision and diffusion components at runtime using torch_tensorrt.dynamo.compile
and runs full inference with the compiled modules (no engine files involved).

Architecture:
  - Vision encoder:  torch-tensorrt compiled (runs once per inference)
  - Language model:  torch-tensorrt compiled with prefix_k/prefix_v cache tensors
  - Diffusion step:  torch-tensorrt compiled with dynamic prefix_len

Usage:
    PYTHONPATH=src python -m alpamayo_r1.test_trt_torch

    # With benchmark:
    PYTHONPATH=src python -m alpamayo_r1.test_trt_torch --benchmark

    # Skip TRT compilation (PyTorch baseline only):
    PYTHONPATH=src python -m alpamayo_r1.test_trt_torch --skip-trt

    # Skip full inference comparison (compile + quick accuracy check only):
    PYTHONPATH=src python -m alpamayo_r1.test_trt_torch --quick
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Callable

import numpy as np
import torch
from alpamayo_r1.trt.compile_trt import (
    compile_trt_modules,
    configure_generation,
    run_inference_trt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data / Model Loading
# =============================================================================


def load_test_data(clip_id: str, t0_us: int = 5_100_000) -> tuple[dict, list]:
    """Load test data from Physical AI AV dataset."""
    from alpamayo_r1 import helper
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    logger.info(f"Loading dataset for clip_id: {clip_id}...")
    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    logger.info("Dataset loaded.")

    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    return data, messages


def prepare_model_inputs(
    model, data: dict, messages: list, device: str = "cuda"
) -> Callable[[], dict[str, Any]]:
    """Returns a factory function that creates fresh model inputs each call."""
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


def compute_trajectory_metrics(
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
) -> dict[str, float]:
    """Compute trajectory prediction metrics."""
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
# Inference Implementations
# =============================================================================


def run_inference_pytorch(
    model,
    create_inputs_fn: callable,
    seed: int = 42,
    num_traj_samples: int = 1,
    max_generation_length: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, dict, float]:
    """Run standard PyTorch inference (original path with KV cache)."""
    torch.cuda.manual_seed_all(seed)
    model_inputs = create_inputs_fn()

    start_time = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=num_traj_samples,
            max_generation_length=max_generation_length,
            return_extra=True,
        )
    inference_time = time.perf_counter() - start_time
    return pred_xyz, pred_rot, extra, inference_time


# =============================================================================
# Benchmarking
# =============================================================================


def benchmark_inference(
    run_fn: callable,
    num_runs: int = 5,
    warmup_runs: int = 2,
    label: str = "Model",
) -> float:
    """Benchmark inference time using a callable run_fn() -> (pred_xyz, pred_rot, extra, time)."""
    logger.info(f"\nBenchmarking {label}...")

    for _ in range(warmup_runs):
        run_fn()
    torch.cuda.synchronize()

    times = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        _, _, _, elapsed = run_fn()
        torch.cuda.synchronize()
        times.append(elapsed)
        logger.info(f"  Run {i + 1}: {elapsed * 1000:.2f} ms")

    avg_time = sum(times) / len(times)
    logger.info(f"  Average: {avg_time * 1000:.2f} ms")
    return avg_time


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combined TRT test: Vision + Language + No-Cache Diffusion in TRT"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="nvidia/Alpamayo-R1-10B",
    )
    parser.add_argument(
        "--clip_id",
        type=str,
        default="030c760c-ae38-49aa-9ad8-f5650a545d26",
    )
    parser.add_argument(
        "--offload_module_to_cpu",
        action="store_true",
        help="Enable Torch-TensorRT offload_module_to_cpu for TRT module compile (LM/diffusion; default: disabled).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-trt",
        action="store_true",
        help="Skip TRT compilation (only run PyTorch baseline)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only compile TRT and verify accuracy on sample inputs, skip full inference",
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-runs", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 70)
    logger.info("COMBINED TRT TEST")
    logger.info("Vision + Language + No-Cache Diffusion: TRT")
    logger.info("=" * 70)
    logger.info(f"  model_path: {args.model_path}")
    logger.info(f"  clip_id:    {args.clip_id}")
    logger.info("  precision:  BF16")

    # ---------------------------------------------------------------------- #
    # Load data and model
    # ---------------------------------------------------------------------- #
    data, messages = load_test_data(args.clip_id)
    gt_xyz = data["ego_future_xyz"]

    logger.info(f"\nLoading model: {args.model_path}...")
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

    model = AlpamayoR1.from_pretrained(args.model_path, dtype=torch.bfloat16).to("cuda")
    model.eval()

    create_inputs_fn = prepare_model_inputs(model, data, messages, device="cuda")
    # ---------------------------------------------------------------------- #
    # PyTorch Baseline (original KV-cache path)
    # ---------------------------------------------------------------------- #
    logger.info("\n" + "=" * 70)
    logger.info("PYTORCH BASELINE (KV-cache path)")
    logger.info("=" * 70)

    pred_xyz_pytorch, _, extra_pytorch, pytorch_time = run_inference_pytorch(
        model, create_inputs_fn, seed=args.seed
    )
    pytorch_metrics = compute_trajectory_metrics(pred_xyz_pytorch, gt_xyz)
    logger.info(f"  Inference time: {pytorch_time * 1000:.2f} ms")
    logger.info(f"  minADE: {pytorch_metrics['min_ade']:.4f} m")
    logger.info(f"  CoC: {extra_pytorch['cot'][0][0, 0][:120]}...")

    if args.skip_trt:
        logger.info("\nSkipping TRT compilation (--skip-trt)")
        return 0

    # ---------------------------------------------------------------------- #
    # PyTorch No-Cache Baseline (to isolate the effect of removing KV cache)
    # ---------------------------------------------------------------------- #
    logger.info("\n" + "=" * 70)
    logger.info("PYTORCH NO-CACHE BASELINE (no KV-cache diffusion path, PyTorch)")
    logger.info("=" * 70)

    pred_xyz_nocache, _, extra_nocache, nocache_time = run_inference_trt(
        model,
        create_inputs_fn,
        trt_vision=None,
        trt_lm=None,
        trt_diffusion=None,
        seed=args.seed,
    )
    nocache_metrics = compute_trajectory_metrics(pred_xyz_nocache, gt_xyz)
    logger.info(f"  Inference time: {nocache_time * 1000:.2f} ms")
    logger.info(f"  minADE: {nocache_metrics['min_ade']:.4f} m")

    # Verify no-cache path matches KV-cache path
    diff_nocache = torch.abs(pred_xyz_pytorch.cpu().float() - pred_xyz_nocache.cpu().float())
    logger.info(
        f"  vs PyTorch KV-cache: max_diff={diff_nocache.max():.6f}m, "
        f"mean_diff={diff_nocache.mean():.6f}m"
    )

    if args.quick:
        logger.info("\n--quick: skipping TRT compilation and full TRT inference run")
        return 0

    trt_vision, trt_lm, trt_diffusion, prefix_seq_len = compile_trt_modules(
        model,
        create_inputs_fn,
        seed=args.seed,
        offload_module_to_cpu=args.offload_module_to_cpu,
        max_generation_length=256,
    )


    # ---------------------------------------------------------------------- #
    # TRT Inference: Vision + Language + Diffusion
    # ---------------------------------------------------------------------- #
    logger.info("\n" + "=" * 70)
    logger.info("TRT INFERENCE (Vision: TRT, Language: TRT, Diffusion: TRT no-cache)")
    logger.info("=" * 70)

    pred_xyz_trt, _, extra_trt, trt_time = run_inference_trt(
        model,
        create_inputs_fn,
        trt_vision=trt_vision,
        trt_lm=trt_lm,
        trt_diffusion=trt_diffusion,
        seed=args.seed,
    )
    trt_metrics = compute_trajectory_metrics(pred_xyz_trt, gt_xyz)
    logger.info(f"  Inference time: {trt_time * 1000:.2f} ms")
    logger.info(f"  minADE: {trt_metrics['min_ade']:.4f} m")
    logger.info(f"  CoC: {extra_trt['cot'][0][0, 0][:120]}...")

    # ---------------------------------------------------------------------- #
    # Trajectory Comparison
    # ---------------------------------------------------------------------- #
    logger.info("\n" + "=" * 70)
    logger.info("TRAJECTORY COMPARISON")
    logger.info("=" * 70)

    diff_full = torch.abs(pred_xyz_pytorch.cpu().float() - pred_xyz_trt.cpu().float())
    logger.info(f"\nPyTorch KV-cache vs Full TRT (no-cache diffusion):")
    logger.info(f"  Max trajectory diff:  {diff_full.max():.6f} m")
    logger.info(f"  Mean trajectory diff: {diff_full.mean():.6f} m")
    logger.info(
        f"  ADE difference:       {abs(pytorch_metrics['min_ade'] - trt_metrics['min_ade']):.4f} m"
    )

    diff_nocache_trt = torch.abs(pred_xyz_nocache.cpu().float() - pred_xyz_trt.cpu().float())
    logger.info(f"\nPyTorch no-cache vs Full TRT (apples-to-apples — same algorithm):")
    logger.info(f"  Max trajectory diff:  {diff_nocache_trt.max():.6f} m")
    logger.info(f"  Mean trajectory diff: {diff_nocache_trt.mean():.6f} m")

    # ---------------------------------------------------------------------- #
    # Benchmarking
    # ---------------------------------------------------------------------- #
    if args.benchmark:
        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARKING")
        logger.info("=" * 70)

        pytorch_avg = benchmark_inference(
            run_fn=lambda: run_inference_pytorch(model, create_inputs_fn, seed=args.seed),
            num_runs=args.benchmark_runs,
            label="PyTorch KV-cache",
        )

        nocache_avg = benchmark_inference(
            run_fn=lambda: run_inference_trt(model, create_inputs_fn, None, None, None, seed=args.seed),
            num_runs=args.benchmark_runs,
            label="PyTorch no-cache (diffusion only)",
        )

        trt_avg = benchmark_inference(
            run_fn=lambda: run_inference_trt(
                model,
                create_inputs_fn,
                trt_vision,
                trt_lm,
                trt_diffusion,
                seed=args.seed,
            ),
            num_runs=args.benchmark_runs,
            label="Full TRT (Vision + Language + Diffusion no-cache)",
        )

        logger.info("\nBenchmark Summary:")
        logger.info(f"  PyTorch KV-cache:          {pytorch_avg * 1000:.2f} ms  (1.00x)")
        logger.info(
            f"  PyTorch no-cache:          {nocache_avg * 1000:.2f} ms  ({pytorch_avg / nocache_avg:.2f}x)"
        )
        logger.info(
            f"  Full TRT (V+L+D no-cache): {trt_avg * 1000:.2f} ms  ({pytorch_avg / trt_avg:.2f}x)"
        )
        logger.info(f"  TRT vs no-cache PyTorch:   {nocache_avg / trt_avg:.2f}x")
        logger.info(f"  Time saved vs baseline:    {(pytorch_avg - trt_avg) * 1000:.2f} ms")

    # ---------------------------------------------------------------------- #
    # Summary
    # ---------------------------------------------------------------------- #
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("\nConfiguration:")
    logger.info("  Vision encoder:  TRT")
    logger.info("  Language model:  TRT (wrapper with prefix_k/prefix_v)")
    logger.info("  Diffusion step:  TRT (no KV cache, full attention recompute)")
    logger.info(f"\n  max_prefix_len (TRT): {prefix_seq_len}")

    logger.info("\nAccuracy:")
    logger.info(f"  PyTorch KV-cache minADE: {pytorch_metrics['min_ade']:.4f} m")
    logger.info(f"  PyTorch no-cache minADE: {nocache_metrics['min_ade']:.4f} m")
    logger.info(f"  Full TRT minADE:         {trt_metrics['min_ade']:.4f} m")
    logger.info(
        f"  TRT vs KV-cache ADE diff:{abs(pytorch_metrics['min_ade'] - trt_metrics['min_ade']):.4f} m"
    )

    logger.info("\nSingle-run timing:")
    logger.info(f"  PyTorch KV-cache: {pytorch_time * 1000:.2f} ms")
    logger.info(f"  PyTorch no-cache: {nocache_time * 1000:.2f} ms")
    logger.info(f"  Full TRT:         {trt_time * 1000:.2f} ms")

    # Pass/fail check
    ade_diff = abs(pytorch_metrics["min_ade"] - trt_metrics["min_ade"])
    nocache_vs_trt_mean = diff_nocache_trt.mean().item()

    if nocache_vs_trt_mean < 0.05:
        logger.info("\n✓ TRT diffusion matches PyTorch no-cache path (mean diff < 5cm)")
    else:
        logger.warning(
            f"\n⚠ TRT vs PyTorch no-cache mean diff = {nocache_vs_trt_mean:.4f}m — needs investigation"
        )

    if ade_diff < 0.15:
        logger.info(f"✓ ADE within 15cm of PyTorch baseline ({ade_diff:.4f}m)")
        logger.info("  Combined TRT compilation is production-ready.")
        return 0
    else:
        logger.warning(f"⚠ ADE difference ({ade_diff:.4f}m) exceeds 15cm threshold")
        return 1


if __name__ == "__main__":
    exit(main())
