import torch
import torch.nn.functional as F
from torch import nn
import torch_tensorrt
from alpamayo_r1.diffusion.flow_matching import FlowMatching
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionAttention,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def check_output_equal(
    output1,
    output2,
    rtol=0.05,
    atol=0.05,
) -> bool:
    if type(output1) != type(output2):
        logger.warning(
            "The output types are different. Check_output_equal will always return false."
        )
        return False

    if isinstance(output1, torch.Tensor):
        if output1.shape != output2.shape:
            return False
        if torch.allclose(output1, output2, rtol, atol):  # type: ignore
            return True
        else:
            print(f"Output {output1} and {output2} are not close. Max diff: {torch.max(torch.abs(output1 - output2))}, Mean diff: {torch.mean(torch.abs(output1 - output2))}")
            return False
    
    elif isinstance(output1, (tuple, list)):
        if len(output1) != len(output2):
            return False
        for a, b in zip(output1, output2):
            if not check_output_equal(a, b, rtol, atol):
                return False
            return True

    elif isinstance(output1, dict):
        if output1.keys() != output2.keys():
            return False
        for a, b in zip(output1.values(), output2.values()):
            if not check_output_equal(a, b, rtol, atol):
                return False
        return True

    logger.warning(
        "The output type is not supported to be checked. Check_output_equal will always return false."
    )
    return False


def plot_error_histogram(
    torch_output,
    trt_output,
    save_path: str = "error_histogram.png",
    bins: int = 100,
    log_scale: bool = True,
):
    """
    Draw histograms of the element-wise absolute error between torch_output
    and trt_output.  Supports torch.Tensor, list/tuple of tensors, or
    dict of tensors (recursively).

    Args:
        torch_output: output from the PyTorch model (Tensor | list | tuple | dict)
        trt_output:   output from the TensorRT model (same structure)
        save_path:    where to save the figure
        bins:         number of histogram bins
        log_scale:    if True, use log scale on the y-axis
    """

    def _flatten_tensors(out, prefix=""):
        """Recursively flatten an output structure into a list of (name, tensor) pairs."""
        results = []
        if isinstance(out, torch.Tensor):
            results.append((prefix or "tensor", out))
        elif isinstance(out, dict):
            for key, val in out.items():
                results.extend(_flatten_tensors(val, prefix=f"{prefix}.{key}" if prefix else str(key)))
        elif isinstance(out, (list, tuple)):
            for i, val in enumerate(out):
                results.extend(_flatten_tensors(val, prefix=f"{prefix}[{i}]"))
        return results

    flat_torch = _flatten_tensors(torch_output, prefix="torch")
    flat_trt = _flatten_tensors(trt_output, prefix="trt")

    if len(flat_torch) != len(flat_trt):
        logger.warning(
            f"plot_error_histogram: torch output has {len(flat_torch)} tensors "
            f"but trt output has {len(flat_trt)} tensors. "
            f"Comparing the first {min(len(flat_torch), len(flat_trt))} pairs."
        )

    error_pairs = []
    for (torch_name, t_tensor), (trt_name, trt_tensor) in zip(flat_torch, flat_trt):
        diff = (t_tensor.float() - trt_tensor.float()).abs().detach().cpu().numpy().ravel()
        label = torch_name.removeprefix("torch")  # e.g. "[0]", ".last_hidden_state"
        error_pairs.append((label or "output", diff))

    if not error_pairs:
        logger.warning("plot_error_histogram: no tensor pairs found to compare.")
        return

    n = len(error_pairs)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), squeeze=False)

    for idx, (name, diff) in enumerate(error_pairs):
        ax = axes[idx, 0]
        ax.hist(diff, bins=bins, edgecolor="black", alpha=0.75)
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Absolute Error")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{name}  |  max={diff.max():.6e}  mean={diff.mean():.6e}  "
            f"median={np.median(diff):.6e}"
        )
        ax.axvline(diff.mean(), color="red", linestyle="--", label="mean")
        ax.axvline(np.median(diff), color="orange", linestyle="--", label="median")
        ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Error histogram saved to {save_path}")


# ====================================================================================================
# 1. Compile model.vlm.model.visual
# ====================================================================================================

_orig_qwen3vl_attn_forward = Qwen3VLVisionAttention.forward

def _qwen3vl_attn_forward_static_lengths(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
) -> torch.Tensor:
    static_lengths = getattr(self, "_static_lengths", None)
    if static_lengths is None or self.config._attn_implementation == "flash_attention_2":
        logger.info("Qwen3VLVisionAttention.forward: using original forward pass")
        return _orig_qwen3vl_attn_forward(
            self,
            hidden_states,
            cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    logger.info("Qwen3VLVisionAttention.forward: using static_lengths forward pass")
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    splits = [torch.split(tensor, static_lengths, dim=2) for tensor in (query_states, key_states, value_states)]
    attn_outputs = [
        attention_interface(
            self,
            q,
            k,
            v,
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            is_causal=False,
            **kwargs,
        )[0]
        for q, k, v in zip(*splits)
    ]
    attn_output = torch.cat(attn_outputs, dim=1)

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output

Qwen3VLVisionAttention.forward = _qwen3vl_attn_forward_static_lengths


class VisualFixedGrid(torch.nn.Module):
    def __init__(self, visual, grid_thw: torch.Tensor):
        super().__init__()
        self.visual = visual.eval()

        logger.info("VisualFixedGrid initialized. Precomputing everything that depends on `grid_thw`")
        with torch.no_grad():
            pos_embeds = self.visual.fast_pos_embed_interpolate(grid_thw)   # (seq, dim)
            rotary_pos_emb = self.visual.rot_pos_emb(grid_thw)              # (seq, dim)

            seq_len = pos_embeds.shape[0]
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos, sin = emb.cos(), emb.sin()

            cu_seqlens = torch.repeat_interleave(
                grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
            ).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
            lengths = torch.repeat_interleave(
                grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
            ).to("cpu").tolist()
            logger.info(f"VisualFixedGrid: static_lengths: {lengths}")
            self._static_lengths = [int(x) for x in lengths]
            for blk in self.visual.blocks:
                blk.attn._static_lengths = self._static_lengths

        self.register_buffer("pos_embeds", pos_embeds, persistent=False)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cu_seqlens", cu_seqlens, persistent=False)

    def forward(self, hidden_states: torch.Tensor, grid_thw=None):
        hidden_states = self.visual.patch_embed(hidden_states)
        torch._check(hidden_states.shape[0] != 0)
        hidden_states = hidden_states + self.pos_embeds.to(hidden_states.dtype)

        position_embeddings = (self.cos.to(hidden_states.dtype), self.sin.to(hidden_states.dtype))

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.visual.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=self.cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.visual.deepstack_visual_indexes:
                deepstack_feature = self.visual.deepstack_merger_list[
                    self.visual.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.visual.merger(hidden_states)
        return hidden_states, deepstack_feature_lists


def compile_qwen3vl_visual(model, model_inputs, plot=False):
    """
    Compile the Qwen3VLVisionModel.
    Args:
        model: Qwen3VLVisionModel. inspect.signature(model.forward): (hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor
        model_inputs: model inputs
    Returns:
        VisualTRTWrapper: VisualTRTWrapper
    """
    model.config.use_cache = False
    logger.info("================== Compiling Qwen3VLVisionModel ==================")
    logger.info(f"model.config:\n{model.config}")
    model.config.attn_implementation = "sdpa"
    model.config._attn_implementation = "sdpa"
    device = torch.device("cuda:0")

    settings = {
        "truncate_double": True,
        "min_block_size": 1,
        "use_python_runtime": True,
        "immutable_weights": True,
        "offload_module_to_cpu": False,
        "use_explicit_typing": False,
        "enabled_precisions": {torch.bfloat16},
        "dryrun": False,
        # "use_fp32_acc": True,
        # "require_full_compilation": True,
    }

    model = model.to(dtype=torch.bfloat16, device=device).eval()

    pixel_values = model_inputs["tokenized_data"]["pixel_values"].to(dtype=torch.bfloat16, device=device)  # (num_images, embed_dim)
    image_grid_thw = model_inputs["tokenized_data"]["image_grid_thw"]  # (num_images, 3)

    inputs = (pixel_values, None)

    wrap = VisualFixedGrid(model, image_grid_thw).to(device).eval()

    try:
        logger.info("Trying to export the model using torch.export.export()..")
        ep = torch.export.export(
            wrap,
            args=inputs,
            # dynamic_shapes=dynamic_shapes,
            strict=False,
        )
    except:
        logger.info("Trying torch.export._trace._export to trace the graph since torch.export.export() failed")
        ep = torch.export._trace._export(
            wrap,
            args=inputs,
            # dynamic_shapes=dynamic_shapes,
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )

    trt_model = torch_tensorrt.dynamo.compile(
        ep,
        inputs,
        **settings
    )

    logger.info("Qwen3VLVisionModel was successfully compiled")

    torch_output = wrap(*inputs)
    trt_output = trt_model(*inputs)
    if check_output_equal(torch_output, trt_output, 0.05, 0.05):
        print("compile_qwen3vl_visual output is correct")
    else:
        print("compile_qwen3vl_visual output is incorrect")

    if plot:
        plot_error_histogram(torch_output, trt_output, save_path="visual_error_histogram.png")

    return trt_model


# ====================================================================================================
# 2. Compile model.vlm.model.language_model
# ====================================================================================================
class LMCompiledWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        **kwargs,
    ):
        # TODO: Need to add cache arguments to the model forward pass if using KV cache
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            # TODO: Enable KV cache
            past_key_values=None,
            cache_position=None,
            # cache_implementation="static",
            # args for deepstack
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        return outputs

def compile_qwen3vl_language_model(model, model_inputs, use_cache=False, cache_implementation="static", plot=False):
    """
    Compile the Qwen3VLTextModel.
    Args:
        model: Qwen3VLTextModel. inspect.signature(model.forward): (input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, past_key_values: Optional[transformers.cache_utils.Cache] = None, inputs_embeds: Optional[torch.FloatTensor] = None, use_cache: Optional[bool] = None, cache_position: Optional[torch.LongTensor] = None, visual_pos_masks: Optional[torch.Tensor] = None, deepstack_visual_embeds: Optional[list[torch.Tensor]] = None, **kwargs: Unpack[transformers.modeling_flash_attention_utils.FlashAttentionKwargs]) -> Union[tuple, transformers.modeling_outputs.BaseModelOutputWithPast]
        settings: compilation settings for torch_tensorrt
    Returns:
        LMCompiledWrapper: LMCompiledWrapper
    """
    logger.info("================== Compiling Qwen3VLTextModel ==================")
    logger.info(f"model.config:\n{model.config}")
    model.config._attn_implementation = "sdpa"
    model.config.attn_implementation = "sdpa"
    settings = {
        "truncate_double": True,
        "min_block_size": 1,
        "use_python_runtime": True,
        "immutable_weights": True,
        "offload_module_to_cpu": False,
        # "enabled_precisions": {torch.bfloat16},
        "use_explicit_typing": True,
        "use_fp32_acc": True,
        "dryrun": False,
    }
    device = torch.device("cuda:0")
    
    model = model.to(dtype=torch.bfloat16, device=device).eval()
    B = 1       # batch
    S = 3006     # sequence length
    hidden_size = model.config.hidden_size

    
    input_ids = None  # model_inputs["tokenized_data"]["input_ids"]
    dtype = torch.bool if use_cache and cache_implementation == "static" else torch.int64
    attention_mask = model_inputs["tokenized_data"]["attention_mask"].to(dtype=dtype, device=device)
    # Qwen3VL passes position_ids shaped (3, B, S)
    # position_ids = torch.arange(S, device=device, dtype=torch.long).view(1, -1).expand(B, -1)
    # position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    position_ids = None
    past_key_values = None  # torch.zeros(B, 2, S, hidden_size, dtype=torch.bfloat16, device=device)
    inputs_embeds = torch.randn(B, S, hidden_size, dtype=torch.bfloat16, device=device)
    use_cache = False
    cache_position = None  # torch.arange(S, dtype=torch.long, device=device)
    visual_pos_masks = None
    deepstack_visual_embeds = None

    # BATCH = torch.export.Dim("batch", min=1, max=32)
    SEQ = torch.export.Dim("seq_len", min=1, max=8192)

    inputs = (input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, cache_position, visual_pos_masks, deepstack_visual_embeds)
    # inputs = (input_ids, attention_mask)
    
    # kw_inputs = {
    #     "input_ids": input_ids,
    #     "attention_mask": attention_mask,
    #     "position_ids": position_ids,
    #     "past_key_values": past_key_values,
    #     "inputs_embeds": inputs_embeds,
    #     "use_cache": use_cache,
    #     "cache_position": cache_position,
    #     "visual_pos_masks": visual_pos_masks,
    #     "deepstack_visual_embeds": deepstack_visual_embeds,
    # }
    dynamic_shapes = {
        "input_ids": None, # {1: SEQ},
        "attention_mask": {1: SEQ},
        "position_ids": None, # {1: BATCH, 2: SEQ},
        "past_key_values": None,
        "inputs_embeds": {1: SEQ},
        "use_cache": None,
        "cache_position": None, # {0: SEQ},
        "visual_pos_masks": None,
        "deepstack_visual_embeds": None,
    }

    lm_wrapper = LMCompiledWrapper(model).to(device).eval()

    # For Torch >= 2.9.0, use the following code
    # try:
    #     logger.info("Trying to export the model using torch.export.export()..")
    #     ep = torch.export.export(
    #         lm_wrapper,
    #         args=(),
    #         kwargs=kw_inputs,
    #         dynamic_shapes=dynamic_shapes,
    #         strict=False,
    #     )
    # except:
    #     logger.info("Trying torch.export._trace._export to trace the graph since torch.export.export() failed")
    #     ep = torch.export._trace._export(
    #         lm_wrapper,
    #         args=(),
    #         kwargs=kw_inputs,
    #         dynamic_shapes=dynamic_shapes,
    #         strict=False,
    #         prefer_deferred_runtime_asserts_over_guards=True,
    #     )

    try:
        logger.info("Trying to export the model using torch.export.export()..")
        ep = torch.export.export(
            lm_wrapper,
            inputs,
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )
    except:
        logger.info("Trying torch.export._trace._export to trace the graph since torch.export.export() failed")
        ep = torch.export._trace._export(
            lm_wrapper,
            inputs,
            dynamic_shapes=dynamic_shapes,
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )

    trt_lm = torch_tensorrt.dynamo.compile(
        ep,
        inputs,
        **settings
    )

    # torch._dynamo.mark_dynamic(attention_mask, index=1, min=1, max=8192)
    # torch._dynamo.mark_dynamic(inputs_embeds, index=1, min=1, max=8192)
    # trt_lm = torch.compile(lm_wrapper, backend="torch_tensorrt", dynamic=dynamic_shapes)

    # with torch_tensorrt.dynamo.Debugger():
    #     trt_out = trt_lm(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, cache_position, visual_pos_masks, deepstack_visual_embeds)
    #     logger.info(f"compile_qwen3vl_language_model trt_out:\n{trt_out}")
    #     logger.info(f"trt_out[0].shape: {trt_out[0].shape}")
    
    print("Lm model was successfully compiled!!!!!!!!!!!!!!!")
    torch_output = lm_wrapper(*inputs)
    trt_output = trt_lm(*inputs)
    if check_output_equal(torch_output, trt_output, 0.05, 0.05):
        print("compile_qwen3vl_language_model output is correct")
    else:
        print("compile_qwen3vl_language_model output is incorrect")

    if plot:
        plot_error_histogram(torch_output, trt_output, save_path="language_error_histogram.png")



    return trt_lm



# Not needed because it can be compiled in the diffusion model
# # 2. Compile model.expert
# class ExpertTRTWrapper(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(
#         self,
#         inputs_embeds=None,
#         position_ids=None,
#         past_key_values=None,
#         attention_mask=None,
#         use_cache=None,
#         is_causal=None,  # accepted but ignored
#         **kwargs,        # ignore any extra kwargs
#     ):
#         return self.model(
#             inputs_embeds=inputs_embeds,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             attention_mask=attention_mask,
#             use_cache=use_cache,
#             is_causal=is_causal,
#         )

# def compile_expert(model, settings):
#     print("================== compile_expert ==================")
#     print("model.config:\n", model.config)
#     print("inspect.signature(model.forward):\n", inspect.signature(model.forward))

#     model = model.to(dtype=torch.bfloat16).eval()
#     device = torch.device("cuda:0")

#     B = 1        # batch
#     S = 128     # sequence length (prompt length)
#     hidden_size = model.config.hidden_size

#     input_ids = None # torch.randint(0, model.config.vocab_size, (B, S), dtype=torch.long, device=device)
#     attention_mask = torch.ones((B, S), dtype=torch.long, device=device)
#     # Qwen3VL passes position_ids shaped (3, B, S)
#     # position_ids = torch.arange(S, device=device, dtype=torch.long).view(1, -1).expand(B, -1)
#     # position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
#     position_ids = None
#     past_key_values = None
#     inputs_embeds = torch.randn(B, S, hidden_size, dtype=torch.bfloat16, device=device)
#     use_cache = False
#     cache_position = torch.arange(S, dtype=torch.long, device=device)
#     visual_pos_masks = None
#     deepstack_visual_embeds = None
#     is_causal = None

#     # BATCH = torch.export.Dim("batch", min=1, max=32)
#     SEQ = torch.export.Dim("seq_len", min=1, max=8192)

#     # inputs = (input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, cache_position, visual_pos_masks, deepstack_visual_embeds, is_causal)
#     inputs = (inputs_embeds, position_ids, past_key_values, attention_mask, use_cache, is_causal)
    
#     dynamic_shapes = {
#         "inputs_embeds": {1: SEQ},
#         "position_ids": None, # {2: SEQ},
#         "past_key_values": None,
#         "attention_mask": {1: SEQ},
#         "use_cache": None,
#         "is_causal": None,
#     }

#     wrapper = ExpertTRTWrapper(model).to(device).eval()

#     try:
#         print("Trying to export the model using torch.export.export()..")
#         ep = torch.export.export(
#             wrapper,
#             args=inputs,
#             dynamic_shapes=dynamic_shapes,
#             strict=False,
#         )
#     except:
#         print("Trying torch.export._trace._export to trace the graph since torch.export.export() failed")
#         ep = torch.export._trace._export(
#             wrapper,
#             args=inputs,
#             dynamic_shapes=dynamic_shapes,
#             strict=False,
#             prefer_deferred_runtime_asserts_over_guards=True,
#         )
    
#     trt_expert = torch_tensorrt.dynamo.compile(
#         ep,
#         inputs,
#         **settings
#     )
#     with torch_tensorrt.dynamo.Debugger():
#         trt_out = trt_expert(inputs_embeds, position_ids, past_key_values, attention_mask, use_cache, is_causal)
#         print("compile_expert trt_out:\n", trt_out)
#         # torch_out = model(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache=use_cache, cache_position=cache_position, visual_pos_masks=visual_pos_masks, deepstack_visual_embeds=deepstack_visual_embeds)
#         # print("compile_expert torch_out:\n", torch_out)
#     return trt_expert


# ====================================================================================================
# 3. Compile model.diffusion
# ====================================================================================================

class FlowMatchingTRTWrapper(torch.nn.Module):
    def __init__(self, flow: FlowMatching, step_fn: torch.nn.Module, inference_steps: int = 10):
        super().__init__()
        self.flow = flow
        self.step_fn = step_fn
        self.inference_steps = inference_steps
    
    def forward(self, x0) -> torch.Tensor:
        """
        It's equivalent to the sample() function in flow matching.
        Args:
            x0: [B, *x_dims]
            inference_steps: int
        Returns:
            x: [B, *x_dims]
        """
        x = x0
        time_steps = torch.linspace(0.0, 1.0, self.inference_steps + 1, device=x.device, dtype=x.dtype)
        n_dim = len(self.flow.x_dims)

        for i in range(self.inference_steps):
            dt = time_steps[i + 1] - time_steps[i]
            dt = dt.view(1, *[1] * n_dim).expand(x.shape[0], *[1] * n_dim)
            t_start = time_steps[i].view(1, *[1] * n_dim).expand(x.shape[0], *[1] * n_dim)
            v = self.step_fn(x, t_start)
            x = x + dt * v
        return x

class StepFnModule(nn.Module):
    def __init__(
        self,
        action_in_proj: nn.Module,
        expert: nn.Module,
        action_out_proj: nn.Module,
        n_diffusion_tokens: int,
        action_space_dims: tuple[int, ...],
    ):
        super().__init__()
        self.action_in_proj = action_in_proj
        self.expert = expert
        self.action_out_proj = action_out_proj
        self.n_diffusion_tokens = n_diffusion_tokens
        self.action_space_dims = action_space_dims

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        # position_ids: torch.Tensor,
        # attention_mask: torch.Tensor,
        # past_key_values,  # tuple/list of tensors (static)
    ) -> torch.Tensor:
        b_star = x.shape[0]
        future_token_embeds = self.action_in_proj(x, t)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(b_star, self.n_diffusion_tokens, -1)

        expert_out = self.expert(
            inputs_embeds=future_token_embeds,
            # position_ids=position_ids,
            # past_key_values=past_key_values,
            # attention_mask=attention_mask,
            use_cache=False,  # disable cache mutation for TRT
        )
        last_hidden = expert_out.last_hidden_state[:, -self.n_diffusion_tokens:]
        pred = self.action_out_proj(last_hidden).view(-1, *self.action_space_dims)
        return pred


class CompiledDiffusion(nn.Module):
    def __init__(self, trt_module: nn.Module, x_dims: tuple[int, ...], dtype: torch.dtype):
        super().__init__()
        self.trt_module = trt_module
        self.x_dims = x_dims
        self.dtype = dtype

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        step_fn=None,  # ignored: baked into TRT module
        device: torch.device | None = None,
        return_all_steps: bool = False,
        *args,
        **kwargs,
    ):
        if return_all_steps:
            raise ValueError("Compiled diffusion does not support return_all_steps.")
        device = device or next(self.trt_module.parameters()).device
        x0 = torch.randn(batch_size, *self.x_dims, device=device, dtype=self.dtype)
        return self.trt_module(x0)


def compile_diffusion(model):
    # model.config.use_cache = False
    print("================== compile_diffusion ==================")
    settings = {
        "truncate_double": True,
        "min_block_size": 1,
        "use_python_runtime": True,
        "immutable_weights": True,
        "offload_module_to_cpu": False,
        "use_explicit_typing": False,
        "enabled_precisions": {torch.bfloat16},
        "dryrun": False,
        # "require_full_compilation": True,
    }
    
    flow = model.diffusion.to(dtype=torch.bfloat16).eval()
    device = torch.device("cuda:0")

    # compile the step_fn
    action_space_dims = model.action_space.get_action_space_dims()
    n_diffusion_tokens = action_space_dims[0]
    step_fn = StepFnModule(model.action_in_proj, model.expert, model.action_out_proj, n_diffusion_tokens, action_space_dims).to(device).eval()
    wrapper = FlowMatchingTRTWrapper(flow, step_fn, inference_steps=10).eval().to(device)

    x0 = torch.randn(1, *flow.x_dims, device=device, dtype=torch.bfloat16)
    ep = torch.export.export(wrapper, args=(x0,), strict=False)

    trt_module = torch_tensorrt.dynamo.compile(
        ep,
        inputs=(x0,),
        **settings,
    )
    with torch_tensorrt.dynamo.Debugger():
        trt_out = trt_module(x0)
        print("compile_diffusion trt_out:\n", trt_out)
        # torch_out = wrapper(x0)
        # print("compile_diffusion torch_out:\n", torch_out)
    return trt_module






# def compile_qwen3vl_model(model, model_inputs, settings):
#     logger.info("================== Compiling Qwen3VLModel ==================")
#     logger.info(f"model.config:\n{model.config}")
#     model.config._attn_implementation = "sdpa"
#     model.config.attn_implementation = "sdpa"

#     model.model.config.text_config.use_cache = False

#     settings = {
#         "truncate_double": True,
#         "min_block_size": 1,
#         "use_python_runtime": True,
#         "immutable_weights": True,
#         "offload_module_to_cpu": False,
#         "use_explicit_typing": False,
#         "enabled_precisions": {torch.bfloat16},
#         "dryrun": False,
#     }
#     device = torch.device("cuda:0")
    
#     model = model.to(dtype=torch.bfloat16, device=device).eval()
#     B = 1       # batch
#     S = 128     # sequence length
#     hidden_size = 4096  # text hidden size

    
#     input_ids = model_inputs["tokenized_data"]["input_ids"]
#     print("111111111111111 input_ids.shape:", input_ids.shape)
#     attention_mask = model_inputs["tokenized_data"]["attention_mask"]
#     # Qwen3VL passes position_ids shaped (3, B, S)
#     # position_ids = torch.arange(S, device=device, dtype=torch.long).view(1, -1).expand(B, -1)
#     # position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
#     position_ids = None
#     past_key_values = None  # torch.zeros(B, 2, S, hidden_size, dtype=torch.bfloat16, device=device)
#     inputs_embeds = None # torch.randn(B, S, hidden_size, dtype=torch.bfloat16, device=device)
    
#     labels = None
#     pixel_values = model_inputs["tokenized_data"]["pixel_values"].to(dtype=torch.bfloat16, device=device)  # (num_images, embed_dim)
#     pixel_values_videos = None
#     image_grid_thw = model_inputs["tokenized_data"]["image_grid_thw"]  # (num_images, 3)
#     video_grid_thw = None
#     cache_position = None
#     logits_to_keep = 0

#     # with torch.no_grad():
#     #     position_ids, rope_deltas = model.model.get_rope_index(
#     #         input_ids,
#     #         image_grid_thw=image_grid_thw,
#     #         video_grid_thw=video_grid_thw,
#     #         attention_mask=attention_mask,
#     #     )
#     #     model.model.rope_deltas = rope_deltas
#     #     print("444444444444444 position_ids.shape:", position_ids.shape)
#     #     print("555555555555555 rope_deltas.shape:", rope_deltas.shape)

#     # use_cache = False
#     # cache_position = None  # torch.arange(S, dtype=torch.long, device=device)
#     # visual_pos_masks = None
#     # deepstack_visual_embeds = None

#     visual_wrapper = VisualFixedGrid(model.model.visual, image_grid_thw).to(device).eval()
#     model.model.visual = visual_wrapper

#     # BATCH = torch.export.Dim("batch", min=1, max=32)
#     SEQ = torch.export.Dim("seq_len", min=1, max=8192)

#     # inputs = (input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, cache_position, visual_pos_masks, deepstack_visual_embeds)
#     kw_inputs = {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "position_ids": position_ids,
#         "past_key_values": past_key_values,
#         "inputs_embeds": inputs_embeds,
#         "labels": labels,
#         "pixel_values": pixel_values,
#         "pixel_values_videos": pixel_values_videos,
#         "image_grid_thw": image_grid_thw,
#         "video_grid_thw": video_grid_thw,
#         "cache_position": cache_position,
#         "logits_to_keep": logits_to_keep,
#     }
#     dynamic_shapes = {
#         "input_ids": None, # {1: SEQ},
#         "attention_mask": {1: SEQ},
#         "position_ids": None, # {2: SEQ},
#         "past_key_values": None,
#         "inputs_embeds": None, # {1: SEQ},
#         "labels": None,
#         "pixel_values": None,
#         "pixel_values_videos": None,
#         "image_grid_thw": None,
#         "video_grid_thw": None,
#         "cache_position": None, # {0: SEQ},
#         "logits_to_keep": None,
#     }

#     # lm_wrapper = LMCompiledWrapper(model).to(device).eval()

#     # try:
#     #     logger.info("Trying to export the model using torch.export.export()..")
#     #     ep = torch.export.export(
#     #         lm_wrapper,
#     #         args=inputs,
#     #         dynamic_shapes=dynamic_shapes,
#     #         strict=False,
#     #     )
#     # except:
#     #     logger.info("Trying torch.export._trace._export to trace the graph since torch.export.export() failed")
    
    
#     ep = torch.export._trace._export(
#         model,
#         args=(),
#         kwargs=kw_inputs,
#         dynamic_shapes=dynamic_shapes,
#         strict=False,
#         prefer_deferred_runtime_asserts_over_guards=True,
#     )

#     trt_model = torch_tensorrt.dynamo.compile(
#         ep,
#         arg_inputs=(),
#         kwarg_inputs=kw_inputs,
#         **settings
#     )

#     logger.info("Qwen3VLModel was successfully compiled")
#     return trt_model
