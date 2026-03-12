#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single-clip demo: PyTorch FP16 vs TRT Plugin FP16.

Runs both pipelines on one clip, prints CoC + minADE, and shows a summary.

For per-engine accuracy comparisons (cosine similarity, max/mean |Δ|),
see ``compare_engines.py``.

Usage:
    python -m alpamayo_r1.demo_trt_plugin
    python -m alpamayo_r1.demo_trt_plugin --num_traj_samples 6

    # With engine cache (skip recompilation on subsequent runs):
    python -m alpamayo_r1.demo_trt_plugin --engine_cache_dir /tmp/trt_cache
"""

import argparse
import gc
import logging
import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "tools", "llm"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import torch

torch._dynamo.config.capture_scalar_outputs = True

from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.trt.plugin_lm import (
    FP16,
    compile_all_engines,
    compute_min_ade,
    load_engines_cache,
    save_engines_cache,
    prepare_model_inputs,
    run_pytorch_inference,
    run_trt_inference,
)


def _hr():
    print("=" * 65)


def main():
    ap = argparse.ArgumentParser(description="Single-clip PyTorch vs TRT Plugin demo")
    ap.add_argument("--clip_id", default="030c760c-ae38-49aa-9ad8-f5650a545d26")
    ap.add_argument("--t0_us", type=int, default=5_100_000)
    ap.add_argument("--ckpt", default="nvidia/Alpamayo-R1-10B")
    ap.add_argument("--max_gen", type=int, default=256)
    ap.add_argument("--num_traj_samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--engine_cache_dir",
        type=str,
        default=None,
        help="Directory for cached TRT engines (.engine format). "
        "If the cache exists, engines are loaded; otherwise compiled and saved.",
    )
    args = ap.parse_args()
    N = args.num_traj_samples

    # ── Load model & data ─────────────────────────────────────
    print("Loading model (FP16)...")
    model = AlpamayoR1.from_pretrained(args.ckpt, dtype=FP16).to("cuda").eval()
    model.config.attn_implementation = "sdpa"

    print(f"Loading data (clip_id={args.clip_id})...")
    data = load_physical_aiavdataset(args.clip_id, t0_us=args.t0_us)

    # ── Compile or load TRT engines (do this FIRST for memory) ─
    _hr()
    engines = None
    if args.engine_cache_dir:
        engines = load_engines_cache(args.engine_cache_dir)
        if engines is not None:
            print(f"✅ Loaded TRT engines from cache: {args.engine_cache_dir}")

    if engines is None:
        print("Compiling TRT engines")
        _hr()
        mi = prepare_model_inputs(model, data)
        engines = compile_all_engines(model, mi, max_gen=args.max_gen, batch_size=N)
        if args.engine_cache_dir:
            save_engines_cache(engines, args.engine_cache_dir)
            print(f"✅ TRT engines saved to cache: {args.engine_cache_dir}")

    torch.cuda.empty_cache(); gc.collect()

    # ── A. PyTorch FP16 ───────────────────────────────────────
    _hr()
    print(f"A. PyTorch FP16 inference (num_traj_samples={N})")
    _hr()

    pred_pt, coc_pt, ms_pt = run_pytorch_inference(
        model, data, max_gen=args.max_gen, num_traj_samples=N, seed=args.seed,
    )
    ade_pt = compute_min_ade(pred_pt, data)
    for i in range(min(N, 3)):
        print(f"  CoC[{i}]: '{coc_pt[i][:70]}...'")
    print(f"  minADE: {ade_pt:.4f} m  ({ms_pt:.0f} ms)")
    torch.cuda.empty_cache(); gc.collect()

    # ── B. TRT Plugin FP16 ────────────────────────────────────
    _hr()
    print(f"B. TRT Plugin FP16 inference (num_traj_samples={N})")
    _hr()

    pred_trt, coc_trt, ms_trt = run_trt_inference(
        model, data, engines, max_gen=args.max_gen, num_traj_samples=N, seed=args.seed,
    )
    ade_trt = compute_min_ade(pred_trt, data)
    for i in range(min(N, 3)):
        print(f"  CoC[{i}]: '{coc_trt[i][:70]}...'")
    print(f"  minADE: {ade_trt:.4f} m  ({ms_trt:.0f} ms)")
    torch.cuda.empty_cache(); gc.collect()

    # ── Summary ───────────────────────────────────────────────
    _hr()
    print("Summary")
    _hr()
    print(f"  num_traj_samples = {N}")
    print(f"  {'Mode':<25} {'minADE':>8}  {'Time':>8}  CoC[0]")
    print(f"  {'-'*65}")
    print(f"  {'PyTorch FP16':<25} {ade_pt:8.4f}  {ms_pt:7.0f}ms  '{coc_pt[0][:35]}...'")
    print(f"  {'TRT Plugin FP16':<25} {ade_trt:8.4f}  {ms_trt:7.0f}ms  '{coc_trt[0][:35]}...'")
    print(f"\n  Δ minADE:  {abs(ade_pt - ade_trt):.4f} m")
    if ms_trt > 0:
        print(f"  Speedup:   {ms_pt / ms_trt:.2f}x  ({ms_pt:.0f} → {ms_trt:.0f} ms)")
    print()


if __name__ == "__main__":
    with torch.no_grad():
        main()
