#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Full-dataset evaluation: PyTorch FP16 vs TRT Plugin FP16.

Evaluates minADE over the Alpamayo gold eval set with batch decode
(all num_traj_samples decoded in parallel).

Uses the same ``compile_all_engines`` / ``run_trt_inference`` pipeline as
``demo_trt_plugin.py``.  Engine caching (``--engine_cache_dir``) is
supported so that repeated evaluations skip the expensive TRT compilation.

Usage:
    # Quick test (3 clips, loading cached engines)
    python -m alpamayo_r1.eval_trt_plugin --limit 3 \\
        --engine_cache_dir /tmp/trt_engine_cache_b6

    # Full evaluation (644 clips)
    python -m alpamayo_r1.eval_trt_plugin --limit 644 \\
        --engine_cache_dir /tmp/trt_engine_cache_b6

    # TRT-only (skip PyTorch, measure TRT memory)
    python -m alpamayo_r1.eval_trt_plugin --limit 644 --skip_pytorch \\
        --engine_cache_dir /tmp/trt_engine_cache_b6

    # PyTorch-only
    python -m alpamayo_r1.eval_trt_plugin --limit 644 --skip_trt
"""

import argparse
import gc
import logging
import os
import subprocess
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "tools", "llm"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch

torch._dynamo.config.capture_scalar_outputs = True

from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
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


# ============================================================
# Utilities
# ============================================================

def read_clip_ids(parquet_path: str) -> list[str]:
    """Read unique clip_ids from parquet file."""
    df = pd.read_parquet(parquet_path)
    cols = {c.lower(): c for c in df.columns}
    ids = df[cols["key"]].astype(str).tolist()
    seen = set()
    return [c for c in ids if not (c in seen or seen.add(c))]


def gpu_mem_smi_gb() -> float:
    """GPU memory used (nvidia-smi, GB)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             f"--id={torch.cuda.current_device()}"], text=True,
        ).strip()
        return int(out) / 1024
    except Exception:
        return 0.0


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="PyTorch FP16 vs TRT Plugin FP16 evaluation"
    )
    ap.add_argument("--parquet", default=str(
        Path(__file__).resolve().parent / "1005_7cam_gold_eval_metadb_public.parquet"))
    ap.add_argument("--t0_us", type=int, default=5_100_000)
    ap.add_argument("--ckpt", default="nvidia/Alpamayo-R1-10B")
    ap.add_argument("--max_gen", type=int, default=256)
    ap.add_argument("--num_traj_samples", type=int, default=6)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42,
                    help="-1 disables per-clip reseeding")
    ap.add_argument("--print_every", type=int, default=25)
    ap.add_argument("--data_dir", default="/data1/physical_ai_av")
    ap.add_argument("--skip_pytorch", action="store_true")
    ap.add_argument("--skip_trt", action="store_true")
    ap.add_argument("--gc_every", type=int, default=1)
    ap.add_argument(
        "--engine_cache_dir",
        type=str,
        default=None,
        help="Directory for cached TRT engines (.engine format). "
        "If the cache exists, engines are loaded; otherwise compiled and saved.",
    )
    args = ap.parse_args()

    clip_ids = read_clip_ids(args.parquet)
    if args.limit > 0:
        clip_ids = clip_ids[:args.limit]
    print(f"Loaded {len(clip_ids)} clips from: {args.parquet}")

    import physical_ai_av
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(local_dir=args.data_dir)
    seed = None if args.seed < 0 else args.seed

    # ── Load model ────────────────────────────────────────────
    print("Loading model (FP16)...")
    model = AlpamayoR1.from_pretrained(args.ckpt, dtype=FP16).to("cuda").eval()
    model.config.attn_implementation = "sdpa"

    # ── Compile or load TRT engines (do this FIRST for memory) ─
    engines = None
    if not args.skip_trt:
        if args.engine_cache_dir:
            engines = load_engines_cache(args.engine_cache_dir)
            if engines is not None:
                print(f"✅ Loaded TRT engines from cache: {args.engine_cache_dir}")

        if engines is None:
            print("Compiling TRT engines...")
            ref_data = load_physical_aiavdataset(
                clip_ids[0], t0_us=args.t0_us, avdi=avdi,
            )
            ref_mi = prepare_model_inputs(model, ref_data)
            engines = compile_all_engines(
                model, ref_mi,
                max_gen=args.max_gen, batch_size=args.num_traj_samples,
            )
            if args.engine_cache_dir:
                save_engines_cache(engines, args.engine_cache_dir)
                print(f"✅ TRT engines saved to cache: {args.engine_cache_dir}")
            del ref_data, ref_mi

        gc.collect(); torch.cuda.empty_cache()

    # ── Evaluate ──────────────────────────────────────────────
    results_pt, results_trt = [], []
    times_pt, times_trt = [], []
    failed = []
    measure_mem = args.skip_pytorch or args.skip_trt
    base_mem = peak_pt = peak_trt = 0.0

    if measure_mem:
        base_mem = gpu_mem_smi_gb()
        print(f"GPU memory baseline: {base_mem:.1f} GB")

    try:
        from tqdm import tqdm
        it = tqdm(clip_ids, desc="Evaluating")
    except ImportError:
        it = clip_ids

    for i, cid in enumerate(it, 1):
        try:
            data = load_physical_aiavdataset(cid, t0_us=args.t0_us, avdi=avdi)

            # PyTorch FP16
            if not args.skip_pytorch:
                pred, _, ms = run_pytorch_inference(
                    model, data, max_gen=args.max_gen,
                    num_traj_samples=args.num_traj_samples, seed=seed,
                )
                ade = compute_min_ade(pred, data)
                results_pt.append(ade); times_pt.append(ms)
                if measure_mem:
                    peak_pt = max(peak_pt, gpu_mem_smi_gb())

            gc.collect(); torch.cuda.empty_cache()

            # TRT Plugin FP16
            if not args.skip_trt:
                pred, _, ms = run_trt_inference(
                    model, data, engines,
                    max_gen=args.max_gen,
                    num_traj_samples=args.num_traj_samples, seed=seed,
                )
                ade = compute_min_ade(pred, data)
                results_trt.append(ade); times_trt.append(ms)
                if measure_mem:
                    peak_trt = max(peak_trt, gpu_mem_smi_gb())

            # Progress
            if args.print_every and i % args.print_every == 0:
                parts = [f"[{i}/{len(clip_ids)}]"]
                if results_pt:
                    parts.append(f"PT={np.mean(results_pt):.4f}m/{np.mean(times_pt):.0f}ms")
                if results_trt:
                    parts.append(f"TRT={np.mean(results_trt):.4f}m/{np.mean(times_trt):.0f}ms")
                print("  ".join(parts))

        except Exception as e:
            failed.append((cid, repr(e)))
            torch.cuda.empty_cache()
            print(f"[{i}] FAILED {cid}: {e}")

        finally:
            if args.gc_every > 0 and i % args.gc_every == 0:
                gc.collect(); torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SUMMARY")
    print("=" * 65)

    if results_pt:
        print(f"  PyTorch FP16:     avg minADE = {np.mean(results_pt):.6f} m  "
              f"({np.mean(times_pt):.1f} ms/clip)  [{len(results_pt)} clips]")
    if results_trt:
        print(f"  TRT Plugin FP16:  avg minADE = {np.mean(results_trt):.6f} m  "
              f"({np.mean(times_trt):.1f} ms/clip)  [{len(results_trt)} clips]")

    if results_pt and results_trt:
        n = min(len(results_pt), len(results_trt))
        diffs = [abs(results_pt[j] - results_trt[j]) for j in range(n)]
        print(f"\n  Per-clip |Δ|:  avg={np.mean(diffs):.6f} m, max={max(diffs):.6f} m")
        print(f"  Avg Δ minADE:  {abs(np.mean(results_pt[:n]) - np.mean(results_trt[:n])):.6f} m")
        if np.mean(times_trt[:n]) > 0:
            su = np.mean(times_pt[:n]) / np.mean(times_trt[:n])
            print(f"  Speedup:       {su:.2f}x  ({np.mean(times_pt[:n]):.0f} → {np.mean(times_trt[:n]):.0f} ms)")

    if measure_mem:
        print(f"\n  GPU Memory (nvidia-smi):")
        print(f"    Baseline:    {base_mem:.1f} GB")
        if peak_pt > 0:
            print(f"    PyTorch:     {peak_pt:.1f} GB  (overhead: ~{peak_pt - base_mem:.1f} GB)")
        if peak_trt > 0:
            print(f"    TRT Plugin:  {peak_trt:.1f} GB  (overhead: ~{peak_trt - base_mem:.1f} GB)")
        print(f"    Current:     {gpu_mem_smi_gb():.1f} GB")

    if failed:
        print(f"\n  Failed: {len(failed)} clips")
        for cid, err in failed[:5]:
            print(f"    {cid}: {err}")

    print("\nDone!")


if __name__ == "__main__":
    with torch.no_grad():
        main()
