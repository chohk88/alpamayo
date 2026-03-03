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

from __future__ import annotations

import logging
import time
from typing import Any

import einops
import numpy as np
import torch
import torch.nn as nn
from transformers import StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList

logger = logging.getLogger(__name__)


def configure_generation(
    model,
    *,
    num_return_sequences: int,
    max_new_tokens: int,
) -> None:
    generation_config = model.vlm.generation_config
    generation_config.top_p = 0.98
    generation_config.temperature = 0.6
    generation_config.do_sample = True
    generation_config.num_return_sequences = num_return_sequences
    generation_config.max_new_tokens = max_new_tokens
    generation_config.output_logits = False
    generation_config.return_dict_in_generate = True
    generation_config.top_k = None
    generation_config.pad_token_id = model.tokenizer.pad_token_id


def compile_vision_trt(
    model: nn.Module,
    model_inputs: dict[str, Any],
    offload_module_to_cpu: bool = False,
) -> nn.Module | None:
    logger.info("\n" + "=" * 60)
    logger.info("Compiling Vision Encoder with TensorRT")
    logger.info("=" * 60)

    from alpamayo_r1.trt.vision import compile_vision_model

    trt_vision = compile_vision_model(
        model.vlm.model.visual,
        model_inputs,
        device="cuda",
        debug=False,
        offload_module_to_cpu=offload_module_to_cpu,
    )
    if trt_vision is None:
        logger.error("Vision TRT compilation failed")
        return None

    logger.info("✓ Vision encoder compiled with TRT")
    return trt_vision


def compile_diffusion_no_cache_trt(
    model: nn.Module,
    prefix_seq_len: int,
    offload_module_to_cpu: bool = False,
) -> nn.Module | None:
    logger.info("\n" + "=" * 60)
    logger.info("Compiling No-Cache Diffusion Step with TensorRT")
    logger.info("=" * 60)

    from alpamayo_r1.trt.diffusion import compile_diffusion_step_no_cache

    compile_diffusion_step_no_cache(
        model,
        max_prefix_len=prefix_seq_len,
        device="cuda",
        offload_module_to_cpu=offload_module_to_cpu,
        debug=False,
        accuracy_check=True,
    )
    if not hasattr(model, "_trt_diffusion_step_no_cache"):
        logger.error("No-cache diffusion TRT compilation failed")
        return None

    logger.info("✓ No-cache diffusion step compiled with TRT")
    return model._trt_diffusion_step_no_cache


def compile_language_trt(
    model: nn.Module,
    max_seq_len: int = 4096,
    max_prefix_len: int | None = None,
    batch_size: int = 1,
    offload_module_to_cpu: bool = False,
) -> nn.Module | None:
    logger.info("\n" + "=" * 60)
    logger.info("Compiling Language Model with TensorRT")
    logger.info("=" * 60)

    from alpamayo_r1.trt.lm_with_cache import compile_vlm_lm_trt_with_cache

    compiled_model = compile_vlm_lm_trt_with_cache(
        model,
        max_seq_len=max_seq_len,
        max_prefix_len=max_seq_len if max_prefix_len is None else max_prefix_len,
        batch_size=batch_size,
        device="cuda",
        offload_module_to_cpu=offload_module_to_cpu,
        debug=False,
        accuracy_check=True,
    )
    model._trt_vlm_backbone = compiled_model
    model._trt_lm_max_batch_size = int(batch_size)
    model._trt_lm_batch_size = int(batch_size)
    logger.info("✓ Language model compiled with TRT")
    return model._trt_vlm_backbone


def measure_prefix_seq_len_for_trt(
    model,
    create_inputs_fn: callable,
    seed: int = 42,
    max_generation_length: int = 256,
) -> int:
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

    configure_generation(model, num_return_sequences=1, max_new_tokens=max_generation_length)
    with torch.no_grad():
        _vlm_out = model.vlm.generate(
            input_ids=_input_ids,
            generation_config=model.vlm.generation_config,
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
    return int(_vlm_out.past_key_values.get_seq_length())


def compile_trt_modules(
    model: nn.Module,
    create_inputs_fn: callable,
    *,
    seed: int = 42,
    offload_module_to_cpu: bool = False,
    max_generation_length: int = 256,
    lm_max_seq_len: int | None = None,
    max_prefix_len: int | None = None,
    num_traj_samples: int = 1,
) -> tuple[nn.Module | None, nn.Module | None, nn.Module | None, int]:
    logger.info("\nMeasuring prefix sequence length for TRT sizing...")
    observed_prefix_seq_len = measure_prefix_seq_len_for_trt(
        model,
        create_inputs_fn,
        seed=seed,
        max_generation_length=max_generation_length,
    )
    prefix_seq_len = observed_prefix_seq_len if max_prefix_len is None else int(max_prefix_len)
    logger.info(f"  observed prefix_seq_len = {observed_prefix_seq_len}")
    logger.info(f"  using max_prefix_len     = {prefix_seq_len}")

    model_inputs = create_inputs_fn()
    trt_vision = None
    # trt_vision = compile_vision_trt(
    #     model,
    #     model_inputs,
    #     offload_module_to_cpu=offload_module_to_cpu,
    # )
    # if trt_vision is None:
    #     logger.error("Failed to compile vision model")

    lm_seq_len = (
        int(prefix_seq_len + max_generation_length)
        if lm_max_seq_len is None
        else int(lm_max_seq_len)
    )
    compile_inputs = create_inputs_fn()
    compile_batch = max(2, int(compile_inputs["ego_history_xyz"].shape[0]) * int(num_traj_samples))
    logger.info(f"  lm compile batch size   = {compile_batch}")

    
    
    trt_lm = None
    trt_lm = compile_language_trt(
        model,
        max_seq_len=lm_seq_len,
        max_prefix_len=lm_seq_len,
        batch_size=compile_batch,
        offload_module_to_cpu=offload_module_to_cpu,
    )
    if trt_lm is None:
        logger.error("Failed to compile language model")


    trt_diffusion = None
    # trt_diffusion = compile_diffusion_no_cache_trt(
    #     model,
    #     prefix_seq_len=prefix_seq_len,
    #     offload_module_to_cpu=offload_module_to_cpu,
    # )
    # if trt_diffusion is None:
    #     logger.error("Failed to compile no-cache diffusion step")


    return trt_vision, trt_lm, trt_diffusion, prefix_seq_len


def run_inference_trt(
    model,
    create_inputs_fn: callable,
    trt_vision: nn.Module | None,
    trt_lm: nn.Module | None,
    trt_diffusion: nn.Module | None,
    seed: int = 42,
    num_traj_samples: int = 1,
    max_generation_length: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, dict, float]:
    """Run inference with TRT vision/language/diffusion modules."""
    from alpamayo_r1.models.alpamayo_r1 import ExpertLogitsProcessor
    from alpamayo_r1.models.token_utils import (
        StopAfterEOS,
        extract_text_tokens,
        replace_padding_after_eos,
        to_special_token,
    )
    from alpamayo_r1.trt.prefix_cache import PrefixKVCache, stack_prefix_kv_from_cache

    torch.cuda.manual_seed_all(seed)
    model_inputs = create_inputs_fn()

    dtype = torch.bfloat16
    device = "cuda"

    start_time = time.perf_counter()

    with torch.autocast("cuda", dtype=dtype):
        ego_history_xyz = model_inputs["ego_history_xyz"]
        ego_history_rot = model_inputs["ego_history_rot"]
        B, _, _, _ = ego_history_xyz.shape
        tokenized_data = model_inputs["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")

        traj_data_vlm = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = model.fuse_traj_tokens(input_ids, traj_data_vlm)

        original_vision_forward = None
        if trt_vision is not None:
            original_vision_forward = model.vlm.model.visual.forward
            model.vlm.model.visual.forward = trt_vision.forward

        eos_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
        configure_generation(
            model,
            num_return_sequences=num_traj_samples,
            max_new_tokens=max_generation_length,
        )

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
            generation_config=model.vlm.generation_config,
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

        b_star = vlm_outputs.sequences.shape[0]
        compiled_max_batch = getattr(
            model,
            "_trt_lm_max_batch_size",
            getattr(model, "_trt_lm_batch_size", None),
        )
        if trt_lm is not None and compiled_max_batch is not None and int(b_star) > int(compiled_max_batch):
            raise ValueError(
                f"TRT LM batch mismatch: compiled max batch={compiled_max_batch}, runtime batch={b_star}. "
                "Recompile TRT LM with a larger num_traj_samples/input batch."
            )
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
        prompt_cache = vlm_outputs.past_key_values
        prefill_seq_len = prompt_cache.get_seq_length()

        position_ids = torch.arange(n_diffusion_tokens, device=device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star).clone()
        delta = vlm_outputs.rope_deltas + offset[:, None]
        position_ids = position_ids + delta.to(device)

        neg_inf = torch.finfo(torch.float32).min
        attention_mask = torch.zeros(
            b_star,
            1,
            n_diffusion_tokens,
            prefill_seq_len + n_diffusion_tokens,
            dtype=torch.float32,
            device=device,
        )
        for i in range(b_star):
            attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = neg_inf
        attention_mask = attention_mask.to(dtype)

        prefix_k, prefix_v = stack_prefix_kv_from_cache(
            prompt_cache,
            device=torch.device(device),
            dtype=dtype,
        )
        prompt_cache = PrefixKVCache(prefix_k, prefix_v)
        if trt_diffusion is not None:
            logger.info(f"  prefix_k shape: {prefix_k.shape}")

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
