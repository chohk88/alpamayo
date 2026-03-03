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
Pure-TensorRT engine save / load / inference utilities.

This module provides two things:

1. **save_trt_engine(engine_bytes, path, metadata)**
   Writes a raw `.trt` engine file and a JSON sidecar `<path>.json` that
   records input/output names, dtypes, and static shapes so the loader
   doesn't need torch_tensorrt at runtime.

2. **TRTEngineRunner**
   A lightweight inference wrapper that loads a `.trt` file using the pure
   TensorRT Python API (``tensorrt`` package only — no torch_tensorrt) and
   exposes a ``__call__(*tensors)`` interface identical to the torch_tensorrt
   compiled modules.

Design
------
* **No torch_tensorrt dependency at inference time.**  The runner imports only
  ``tensorrt`` and ``torch``.
* **Dynamic shapes** are handled via ``context.set_input_shape()`` before each
  ``execute_async_v3`` call, so a single engine compiled with min/opt/max
  profiles handles variable-length inputs.
* **Output allocation** is done lazily on the first call and cached; if an
  output shape changes (dynamic output), tensors are re-allocated.
* The runner is intentionally minimal — no CUDA Graphs, no weight streaming.
  For production you can layer those on top.

Typical workflow
----------------
Compilation side (requires torch_tensorrt):

    from alpamayo_r1.trt.vision  import save_vision_engine
    from alpamayo_r1.trt.diffusion import save_diffusion_engine

    save_vision_engine(exported_program, inputs, "/tmp/vision.trt", precision="BF16")
    save_diffusion_engine(exported_program, trt_input_specs, "/tmp/diffusion.trt",
                          precision="BF16", min_prefix_len=1, max_prefix_len=4096)

Inference side (requires only tensorrt + torch):

    from alpamayo_r1.trt.engine_io import TRTEngineRunner

    runner = TRTEngineRunner("/tmp/vision.trt")
    output_tensors = runner(pixel_values)
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Sequence

import torch

logger = logging.getLogger(__name__)

# Map TensorRT dtype enum values to torch dtypes (populated lazily)
_TRT_TO_TORCH_DTYPE: dict[Any, torch.dtype] | None = None


def _trt_dtype_to_torch(trt_dtype) -> torch.dtype:
    """Convert a tensorrt.DataType to the corresponding torch.dtype."""
    import tensorrt as trt

    global _TRT_TO_TORCH_DTYPE
    if _TRT_TO_TORCH_DTYPE is None:
        _TRT_TO_TORCH_DTYPE = {
            trt.DataType.FLOAT:  torch.float32,
            trt.DataType.HALF:   torch.float16,
            trt.DataType.BF16:   torch.bfloat16,
            trt.DataType.INT8:   torch.int8,
            trt.DataType.INT32:  torch.int32,
            trt.DataType.INT64:  torch.int64,
            trt.DataType.BOOL:   torch.bool,
        }
    return _TRT_TO_TORCH_DTYPE.get(trt_dtype, torch.float32)


def save_trt_engine(
    engine_bytes: bytes,
    path: str | pathlib.Path,
    metadata: dict | None = None,
) -> None:
    """
    Write a serialized TRT engine to disk along with a JSON sidecar.

    Args:
        engine_bytes: Raw bytes from
            ``torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine``
        path:         Destination path for the ``.trt`` file.  The JSON sidecar
                      is written to ``<path>.json``.
        metadata:     Optional dict of metadata to embed in the sidecar
                      (e.g. input/output names, shapes, precision).  If
                      ``None`` an empty sidecar is written.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        f.write(engine_bytes)
    logger.info(f"TRT engine written to {path} ({len(engine_bytes) / 1024 / 1024:.1f} MB)")

    sidecar = path.with_suffix(path.suffix + ".json")
    with open(sidecar, "w") as f:
        json.dump(metadata or {}, f, indent=2)
    logger.info(f"Engine metadata written to {sidecar}")


def load_trt_engine_metadata(path: str | pathlib.Path) -> dict:
    """Load the JSON sidecar for a ``.trt`` file."""
    sidecar = pathlib.Path(str(path) + ".json")
    if sidecar.exists():
        with open(sidecar) as f:
            return json.load(f)
    return {}


class TRTEngineRunner:
    """
    Pure-TensorRT inference wrapper for a serialized ``.trt`` engine file.

    Loads the engine using ``tensorrt.Runtime`` (no torch_tensorrt required)
    and exposes a ``__call__(*input_tensors) -> list[torch.Tensor]`` interface.

    Dynamic shapes are supported: call ``set_input_shape(name, shape)`` or pass
    tensors whose shapes differ from the opt profile — the runner will call
    ``context.set_input_shape`` automatically for every input marked dynamic in
    the engine.

    Args:
        path:       Path to the ``.trt`` engine file.
        device:     CUDA device string (default ``"cuda"``).
        stream:     Optional pre-created ``torch.cuda.Stream``. If ``None``,
                    the current CUDA stream is used.

    Example::

        runner = TRTEngineRunner("/tmp/vision.trt")
        [hidden, *deepstack] = runner(pixel_values)
    """

    def __init__(
        self,
        path: str | pathlib.Path,
        device: str = "cuda",
        stream: torch.cuda.Stream | None = None,
    ):
        import tensorrt as trt

        self.path = pathlib.Path(path)
        self.device = torch.device(device)
        self._stream = stream
        self.metadata = load_trt_engine_metadata(self.path)

        logger.info(f"Loading TRT engine from {self.path}")
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(self.path, "rb") as f:
            engine_bytes = f.read()
        self.engine: trt.ICudaEngine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine from {self.path}")

        self.context: trt.IExecutionContext = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TRT execution context")

        # Separate input and output tensor names (TRT >= 8.5 unified I/O API)
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        logger.info(
            f"  {len(self.input_names)} inputs: {self.input_names}\n"
            f"  {len(self.output_names)} outputs: {self.output_names}"
        )

        # Cache for pre-allocated output tensors; keyed by output shape tuple
        self._output_cache: dict[str, torch.Tensor] = {}

    @property
    def _cuda_stream(self) -> int:
        """Return the raw CUDA stream handle as an int."""
        if self._stream is not None:
            return self._stream.cuda_stream
        return torch.cuda.current_stream(self.device).cuda_stream

    def _get_output_tensor(self, name: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Return a cached output tensor, re-allocating if shape changed."""
        cached = self._output_cache.get(name)
        if cached is None or cached.shape != torch.Size(shape) or cached.dtype != dtype:
            self._output_cache[name] = torch.empty(shape, dtype=dtype, device=self.device)
        return self._output_cache[name]

    def __call__(self, *inputs: torch.Tensor) -> list[torch.Tensor]:
        """
        Run TRT inference.

        Args:
            *inputs: Input tensors in the same order as ``self.input_names``.
                     Tensors must already be on the correct CUDA device.

        Returns:
            List of output tensors, one per ``self.output_names`` entry.
        """
        import tensorrt as trt

        if len(inputs) != len(self.input_names):
            raise ValueError(
                f"Expected {len(self.input_names)} inputs, got {len(inputs)}"
            )

        # Bind inputs and set dynamic shapes
        for name, tensor in zip(self.input_names, inputs):
            tensor = tensor.contiguous().to(self.device)
            self.context.set_tensor_address(name, tensor.data_ptr())
            # Always set shape (harmless for static dims, required for dynamic dims)
            self.context.set_input_shape(name, tuple(tensor.shape))

        # Allocate outputs based on inferred shapes after inputs are bound
        output_tensors: list[torch.Tensor] = []
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            out = self._get_output_tensor(name, shape, dtype)
            self.context.set_tensor_address(name, out.data_ptr())
            output_tensors.append(out)

        # Execute
        self.context.execute_async_v3(self._cuda_stream)
        torch.cuda.current_stream(self.device).synchronize()

        return output_tensors
