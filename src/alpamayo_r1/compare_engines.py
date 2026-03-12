#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Per-engine accuracy comparison: PyTorch vs TRT Plugin (batch_size=1).

Compares each engine component (Vision, LM prefill, Diffusion step)
individually, printing cosine similarity, max |Δ|, and mean |Δ|.

Uses the same ``compile_all_engines`` function as the demo script,
always with ``batch_size=1`` for consistent comparison.  Optionally
loads / saves cached engines via ``--engine_cache_dir``.

Usage:
    python -m alpamayo_r1.compare_engines
    python -m alpamayo_r1.compare_engines --engine_cache_dir /tmp/trt_cache_b1
    python -m alpamayo_r1.compare_engines --clip_id <UUID>
"""

import argparse
import copy
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
    DEVICE,
    PluginWrapperDSInput,
    compile_all_engines,
    load_engines_cache,
    save_engines_cache,
    prepare_model_inputs,
    run_vlm_preprocessing,
    tensor_diff,
)


def _hr():
    print("=" * 65)


def _print_diff(label: str, d: dict):
    print(
        f"  {label:<14}  cos_sim={d['cosine_sim']:.6f}  "
        f"max|Δ|={d['max_diff']:.6f}  mean|Δ|={d['mean_diff']:.6f}"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Per-engine accuracy comparison (batch_size=1)"
    )
    ap.add_argument("--clip_id", default="030c760c-ae38-49aa-9ad8-f5650a545d26")
    ap.add_argument("--t0_us", type=int, default=5_100_000)
    ap.add_argument("--ckpt", default="nvidia/Alpamayo-R1-10B")
    ap.add_argument("--max_gen", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--engine_cache_dir",
        type=str,
        default=None,
        help="Directory for cached TRT engines (batch=1). "
        "If the cache exists, engines are loaded; otherwise compiled and saved.",
    )
    args = ap.parse_args()

    # ── Load model & data ─────────────────────────────────────
    print("Loading model (FP16)...")
    model = AlpamayoR1.from_pretrained(args.ckpt, dtype=FP16).to("cuda").eval()
    model.config.attn_implementation = "sdpa"

    print(f"Loading data (clip_id={args.clip_id})...")
    data = load_physical_aiavdataset(args.clip_id, t0_us=args.t0_us)

    # ── Compile or load engines (always batch_size=1) ─────────
    engines = None
    if args.engine_cache_dir:
        engines = load_engines_cache(args.engine_cache_dir)
        if engines is not None:
            print(f"✅ Loaded engines from cache: {args.engine_cache_dir}")

    if engines is None:
        _hr()
        print("Compiling TRT engines (batch_size=1)")
        _hr()
        mi = prepare_model_inputs(model, data)
        engines = compile_all_engines(model, mi, max_gen=args.max_gen, batch_size=1)
        if args.engine_cache_dir:
            save_engines_cache(engines, args.engine_cache_dir)
            print(f"✅ Engines saved to cache: {args.engine_cache_dir}")

    torch.cuda.empty_cache()
    gc.collect()

    # ── Per-engine accuracy comparisons ───────────────────────
    _hr()
    print("Per-engine accuracy comparison (PyTorch vs TRT, batch_size=1)")
    _hr()

    mi1 = prepare_model_inputs(model, data)
    mi2 = prepare_model_inputs(model, data)

    # --- Vision ---
    with torch.no_grad():
        _, emb_pt, ds_pt, vm_pt, _, _ = run_vlm_preprocessing(
            model, mi1, trt_vision=None
        )
        _, emb_trt, ds_trt, _, _, _ = run_vlm_preprocessing(
            model, mi2, trt_vision=engines["trt_vision"]
        )

    print("\n  [Vision encoder]")
    _print_diff("embeds", tensor_diff(emb_pt, emb_trt))
    for i in range(len(ds_pt)):
        _print_diff(f"deepstack[{i}]", tensor_diff(ds_pt[i], ds_trt[i]))

    # --- LM prefill (batch=1) ---
    from plugin_utils import (
        PluginAttention,
        create_kv_caches,
        set_plugin_config_from_model,
    )

    lm_cfg = engines["lm_config"]
    S = engines["S_input"]
    msl = engines["max_seq_len"]
    nds = engines["num_ds_layers"]

    ds_stack = torch.zeros(
        nds, 1, msl, lm_cfg.hidden_size, dtype=FP16, device=DEVICE
    )
    vp = vm_pt[0].nonzero(as_tuple=True)[0]
    for i in range(nds):
        ds_stack[i, 0, vp, :] = ds_pt[i].to(FP16).squeeze(0)

    embeds_fp16 = emb_pt.to(FP16)
    ctx = torch.tensor([S], dtype=torch.int32, device=DEVICE)

    # Build PyTorch plugin wrapper for LM comparison
    lm_ref = model.vlm.model.language_model
    lm_pt_copy = copy.deepcopy(lm_ref).to(dtype=FP16, device=DEVICE).eval()

    with torch.no_grad():
        mi_p = prepare_model_inputs(model, data)
        _, _, _, _, pos_ids_p, rd_p = run_vlm_preprocessing(
            model, mi_p, trt_vision=None
        )
        d_eff = torch.arange(S, msl, device=DEVICE).float()
        d_eff += rd_p.to(DEVICE).float().squeeze()
        d_3d = d_eff.view(1, 1, -1).expand(3, 1, -1).long()
        full_pos = torch.cat([pos_ids_p.to(DEVICE), d_3d], dim=2)
        cos, sin = lm_pt_copy.rotary_emb(
            torch.ones(1, device=DEVICE, dtype=FP16), full_pos
        )
        h2 = lm_cfg.head_dim // 2
        rope_cache = torch.cat(
            [cos[:, :msl, :h2].float(), sin[:, :msl, :h2].float()], dim=-1
        )

    set_plugin_config_from_model(lm_cfg, msl)
    for i, layer in enumerate(lm_pt_copy.layers):
        layer.self_attn = PluginAttention(layer.self_attn, lm_cfg, i, rope_cache)
    head_pt = copy.deepcopy(model.vlm.lm_head).to(dtype=FP16, device=DEVICE).eval()
    wrapper_pt = PluginWrapperDSInput(lm_pt_copy, head_pt, nds).to(DEVICE).eval()

    kvs_pt = create_kv_caches(lm_cfg, msl, 1, DEVICE, FP16)
    with torch.no_grad():
        pt_logits, _ = wrapper_pt(embeds_fp16, kvs_pt, ctx, ds_stack)

    kvs_trt = create_kv_caches(lm_cfg, msl, 1, DEVICE, FP16)
    with torch.no_grad():
        trt_logits, _ = engines["trt_lm"](embeds_fp16, kvs_trt, ctx, ds_stack)
    del kvs_trt

    print("\n  [LM prefill logits (PyTorch plugin vs TRT plugin)]")
    _print_diff("logits", tensor_diff(pt_logits, trt_logits))
    del trt_logits

    del lm_pt_copy, wrapper_pt, head_pt, kvs_pt, pt_logits
    torch.cuda.empty_cache()
    gc.collect()

    # --- Diffusion step ---
    print("\n  [Diffusion step (batch=1)]")
    from alpamayo_r1.trt.diffusion import _build_diffusion_module

    diff_pt_module, _ = _build_diffusion_module(model, dtype=FP16, device="cuda")
    nd = model.action_space.get_action_space_dims()[0]
    kv_len = S + 20
    nl = lm_cfg.num_hidden_layers
    nkv = lm_cfg.num_key_value_heads
    hd = lm_cfg.head_dim

    pk = torch.randn(nl, 1, nkv, kv_len, hd, dtype=FP16, device=DEVICE) * 0.01
    pv = torch.randn(nl, 1, nkv, kv_len, hd, dtype=FP16, device=DEVICE) * 0.01
    pid = (
        torch.arange(nd, device=DEVICE)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(3, 1, -1)
        .clone()
        + kv_len
    )
    am = torch.zeros(1, 1, nd, kv_len + nd, dtype=FP16, device=DEVICE)

    x_test = torch.randn(
        1, *model.action_space.get_action_space_dims(), dtype=FP16, device=DEVICE
    )
    t_test = torch.zeros(1, 1, 1, dtype=FP16, device=DEVICE)

    with torch.no_grad():
        out_pt_d = diff_pt_module(x_test, t_test, pk, pv, pid, am)
        out_trt_d = engines["trt_diffusion"](x_test, t_test, pk, pv, pid, am)
    _print_diff("step output", tensor_diff(out_pt_d, out_trt_d))

    del diff_pt_module, pk, pv
    torch.cuda.empty_cache()
    gc.collect()

    print("\nDone!")


if __name__ == "__main__":
    with torch.no_grad():
        main()
