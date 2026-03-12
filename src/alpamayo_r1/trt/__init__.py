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
TensorRT compilation subpackage for Alpamayo.

All components use torch.export + torch_tensorrt.dynamo.compile (eager, no lazy
recompile). MTTM (MutableTorchTensorRTModule) is not used.

Submodules
----------
alpamayo_r1.trt.vision
    Compile the Qwen3VL vision encoder.
    Entry points: compile_vision_model(), compile_and_replace_vision_model(),
                  save_vision_engine()

alpamayo_r1.trt.lm
    Compile the Qwen3VL language model backbone with static KV cache
    (static_cache_v2 FX pass).
    Entry points: compile_vlm_lm_trt(), generate_alpamayo_with_static_cache()

alpamayo_r1.trt.diffusion
    Compile the fused diffusion denoising step (action_in_proj + expert + action_out_proj)
    with a dynamic VLM KV prefix length (min/max bounds set at compile time).
    Entry points: compile_diffusion_step_no_cache(), save_diffusion_engine()

alpamayo_r1.trt.engine_io
    Pure-TRT engine serialization and inference (no torch_tensorrt at runtime).
    Entry points: save_trt_engine(), TRTEngineRunner
"""

from alpamayo_r1.trt.diffusion import (
    compile_diffusion_step_no_cache,
    save_diffusion_engine,
)
from alpamayo_r1.trt.engine_io import (
    TRTEngineRunner,
    save_trt_engine,
)
from alpamayo_r1.trt.lm import (
    compile_vlm_lm_trt,
    generate_alpamayo_with_static_cache,
)
from alpamayo_r1.trt.vision import (
    compile_and_replace_vision_model,
    compile_vision_model,
    save_vision_engine,
)
from alpamayo_r1.trt.plugin_lm import (
    compile_all_engines,
    load_engines_cache,
    save_engines_cache,
    run_trt_inference,
    run_pytorch_inference,
)

__all__ = [
    # Vision
    "compile_vision_model",
    "compile_and_replace_vision_model",
    "save_vision_engine",
    # LM
    "compile_vlm_lm_trt",
    "generate_alpamayo_with_static_cache",
    # Diffusion
    "compile_diffusion_step_no_cache",
    "save_diffusion_engine",
    # Engine I/O
    "save_trt_engine",
    "TRTEngineRunner",
    # Plugin LM
    "compile_all_engines",
    "load_engines_cache",
    "save_engines_cache",
    "run_trt_inference",
    "run_pytorch_inference",
]
