import unittest
from dataclasses import dataclass

import torch
from parameterized import parameterized
from transformers.models.gemma3.modeling_gemma3 import \
    Gemma3ForCausalLM as HfGemmaForCausalLM
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextConfig
from utils.util import getSMVersion

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_gemma3 import Gemma3ForCausalLM
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping

# Path to the local Gemma 3 1B model directory
GEMMA3_MODEL_PATH = "/home/scratch.trt_llm_data/llm-models/gemma/gemma-3-1b-it/"

# Debug flag - enable detailed logging
DEBUG_INTERMEDIATE_OUTPUTS = False
SAVE_DEBUG_TENSORS = False  # Set to True to save tensors to files


def debug_log_hf_tensor(name: str, tensor: torch.Tensor, layer_idx: int = None):
    """Log tensor statistics for HF model debugging"""
    if not DEBUG_INTERMEDIATE_OUTPUTS:
        return

    layer_prefix = f"Layer-{layer_idx}: " if layer_idx is not None else ""
    print(f"[HF DEBUG] {layer_prefix}{name}")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Mean: {tensor.float().mean().item():.6f}")
    print(f"  Std: {tensor.float().std().item():.6f}")
    print(f"  Min: {tensor.float().min().item():.6f}")
    print(f"  Max: {tensor.float().max().item():.6f}")
    print(f"  First few values: {tensor.flatten()[:5].float()}")
    print(f"  Last few values: {tensor.flatten()[-5:].float()}")

    # Optionally save tensor to file
    if SAVE_DEBUG_TENSORS:
        import os
        os.makedirs("debug_tensors", exist_ok=True)
        safe_name = name.replace(" ", "_").replace("/", "_").replace(":", "_")
        layer_str = f"_layer_{layer_idx}" if layer_idx is not None else ""
        filename = f"debug_tensors/hf_{safe_name}{layer_str}.pt"
        torch.save(tensor.detach().cpu(), filename)
        print(f"  Saved to: {filename}")

    print()


def compare_tensors(name: str,
                    test_tensor: torch.Tensor,
                    ref_tensor: torch.Tensor,
                    atol: float = 1e-2,
                    rtol: float = 1e-2,
                    layer_idx: int = None) -> bool:
    """Compare two tensors and log differences"""
    if not DEBUG_INTERMEDIATE_OUTPUTS:
        return

    layer_prefix = f"Layer-{layer_idx}: " if layer_idx is not None else ""
    print(f"[COMPARISON] {layer_prefix}{name}")

    # Ensure same shape
    if test_tensor.shape != ref_tensor.shape:
        print(f"  Shape mismatch: {test_tensor.shape} vs {ref_tensor.shape}")
        return False

    # Calculate differences with epsilon to avoid division by zero
    # Choose epsilon based on the original tensor dtype
    if ref_tensor.dtype == torch.float16:
        eps = 1e-4
    elif ref_tensor.dtype == torch.bfloat16:
        eps = 1e-3
    elif ref_tensor.dtype == torch.float32:
        eps = 1e-8
    else:
        # Default epsilon for other dtypes
        eps = 1e-6

    # Convert to float and add epsilon to denominator to prevent division by zero
    test_float = test_tensor.float()
    ref_float = ref_tensor.float()
    rel_diff = (test_float - ref_float) / (abs(ref_float) + eps)
    abs_diff = torch.abs(rel_diff)

    # Find the positions of 10 maximum relative differences
    top_k = min(10, abs_diff.numel())  # Don't exceed tensor size
    top_k_values, top_k_flat_indices = torch.topk(abs_diff.flatten(), top_k)

    # Convert flat indices to multi-dimensional positions
    top_k_positions = [
        torch.unravel_index(idx, abs_diff.shape) for idx in top_k_flat_indices
    ]

    max_diff_value = top_k_values[0].item()  # First value is the maximum
    max_diff_position = top_k_positions[0]  # First position is the max position

    print(f"  Max relative difference: {max_diff_value:.6f}")
    print(f"  Max difference position: {max_diff_position}")
    print(f"  Mean relative difference: {abs_diff.mean().item():.6f}")
    print(f"  Std of relative differences: {abs_diff.std().item():.6f}")

    # Show top 3 differences for more insight
    if DEBUG_INTERMEDIATE_OUTPUTS and top_k > 1:
        print(f"  Top {min(3, top_k)} largest differences:")
        for i in range(min(3, top_k)):
            pos = top_k_positions[i]
            val = top_k_values[i].item()
            test_val = test_tensor[pos].item()
            ref_val = ref_tensor[pos].item()
            print(
                f"    #{i+1}: {val:.6f} at {pos} (Test: {test_val:.6f}, Ref: {ref_val:.6f})"
            )

    # Show the actual values at the max difference position
    if len(max_diff_position) > 0:
        test_val = test_tensor[max_diff_position].item()
        ref_val = ref_tensor[max_diff_position].item()
        print(
            f"  At max diff position={max_diff_position} - Test: {test_val:.6f}, Ref: {ref_val:.6f}"
        )

    # Check if tensors are close
    try:
        torch.testing.assert_close(test_tensor,
                                   ref_tensor,
                                   atol=atol,
                                   rtol=rtol)
        print(f"  ✓ Tensors are close (atol={atol}, rtol={rtol})")
        return True
    except AssertionError:
        print(f"  ✗ Tensors differ significantly (atol={atol}, rtol={rtol})")
        print(f" test_tensor: {test_tensor}")
        print("=" * 80)
        print(f" ref_tensor: {ref_tensor}")

        return False


def enable_debug_logging(save_tensors=False):
    """Enable debug logging for both models"""
    global DEBUG_INTERMEDIATE_OUTPUTS, SAVE_DEBUG_TENSORS
    DEBUG_INTERMEDIATE_OUTPUTS = True
    SAVE_DEBUG_TENSORS = save_tensors

    # Also enable in the TRT-LLM model
    import tensorrt_llm._torch.models.modeling_gemma3 as gemma3_module
    gemma3_module.DEBUG_INTERMEDIATE_OUTPUTS = True
    gemma3_module.SAVE_DEBUG_TENSORS = save_tensors


@dataclass(repr=False)
class Scenario:
    backend: str
    use_cuda_graph: bool = False

    def __repr__(self) -> str:
        return f"backend:{self.backend.lower()}-use_cuda_graph:{self.use_cuda_graph}"


class TestGemma3(unittest.TestCase):

    @parameterized.expand(
        [
            Scenario(backend="TRTLLM"),
            # Scenario(backend="TRTLLM", use_cuda_graph=True),
        ],
        lambda testcase_func, param_num, param:
        f"{testcase_func.__name__}[{param.args[0]}]")
    @torch.no_grad()
    def test_gemma3_allclose_to_hf(self, scenario: Scenario) -> None:
        """
        Compare output to HF
        """
        if getSMVersion() < 80:
            self.skipTest("TRTLLM backend is not supported in pre-Ampere.")

        # Enable debug logging - set to True to see detailed intermediate outputs
        # Change this to True when you want to debug
        if True:  # Set to True to enable debug logging
            enable_debug_logging(save_tensors=False)
            print("=" * 80)
            print("DEBUG LOGGING ENABLED - Comparing intermediate outputs")
            print("=" * 80)

        backend = scenario.backend
        metadata_cls = get_attention_backend(backend).Metadata

        torch.random.manual_seed(0)

        # Dictionary to store HF intermediate outputs
        hf_intermediates = {}

        # Load the actual Gemma 3 1B config from the local model directory
        hf_gemma_config = Gemma3TextConfig.from_pretrained(GEMMA3_MODEL_PATH)
        # Use the model's native dtype
        if hasattr(hf_gemma_config,
                   'torch_dtype') and hf_gemma_config.torch_dtype is not None:
            dtype = hf_gemma_config.torch_dtype
        else:
            dtype = torch.bfloat16  # Default to bfloat16 for Gemma 3 1B
        device = torch.device('cuda')

        # Load the actual Gemma 3 1B model with weights from the local directory
        hf_gemma = HfGemmaForCausalLM.from_pretrained(
            GEMMA3_MODEL_PATH, torch_dtype=dtype).to(device).eval()

        # Add hooks to capture HF intermediate outputs
        def create_pre_hook(name):

            def hook(module,
                     input):  # Note: no 'output' parameter for pre-hooks
                if DEBUG_INTERMEDIATE_OUTPUTS:
                    if isinstance(input, tuple) and len(input) > 0:
                        tensor_input = input[0]
                    else:
                        tensor_input = input
                    # copy the tensor to avoid modifying the original tensor
                    tensor_input = tensor_input.clone()
                    # remove batch dimension
                    if tensor_input.dim() > 2:
                        tensor_input = tensor_input.squeeze(0)
                    hf_intermediates[name] = tensor_input.detach()
                    # debug_log_hf_tensor(f"HF {name}", tensor_input)

            return hook

        def create_post_hook(name):

            def hook(module, input, output):  # Both input and output available
                if DEBUG_INTERMEDIATE_OUTPUTS:
                    if isinstance(output, tuple):
                        tensor_output = output[0]
                    else:
                        tensor_output = output
                    # copy the tensor to avoid modifying the original tensor
                    tensor_output = tensor_output.clone()
                    if tensor_output.dim() > 2:
                        tensor_output = tensor_output.squeeze(0)
                    # if module is q_norm, we need to reshape and get [seq, num_heads*head_dim]
                    if "self_attn_q_norm" in name or "self_attn_k_norm" in name:
                        # tensor_output shape here is [num_heads, seq_len, head_dim]
                        seq_len = tensor_output.shape[1]
                        tensor_output = tensor_output.permute(1, 0, 2).reshape(
                            seq_len, -1)

                    hf_intermediates[name] = tensor_output.detach()

            return hook

        # Register hooks for embedding and layers
        hf_gemma.model.embed_tokens.register_forward_hook(
            create_post_hook("embedding"))
        for i, layer in enumerate(hf_gemma.model.layers):
            # Capture input to layer norm
            layer.input_layernorm.register_forward_pre_hook(
                create_pre_hook(f"layer_{i}_input_layernorm_pre"))

            # Capture output from layer norm
            layer.input_layernorm.register_forward_hook(
                create_post_hook(f"layer_{i}_input_layernorm"))
            layer.self_attn.q_norm.register_forward_hook(
                create_post_hook(f"layer_{i}_self_attn_q_norm"))
            layer.self_attn.k_norm.register_forward_hook(
                create_post_hook(f"layer_{i}_self_attn_k_norm"))
            layer.self_attn.register_forward_hook(
                create_post_hook(f"layer_{i}_self_attention"))
            layer.post_attention_layernorm.register_forward_hook(
                create_post_hook(f"layer_{i}_post_attention_layernorm"))
            layer.pre_feedforward_layernorm.register_forward_hook(
                create_post_hook(f"layer_{i}_pre_feedforward_layernorm"))
            layer.mlp.register_forward_hook(create_post_hook(f"layer_{i}_mlp"))
            layer.post_feedforward_layernorm.register_forward_hook(
                create_post_hook(f"layer_{i}_post_feedforward_layernorm"))
        hf_gemma.model.norm.register_forward_hook(
            create_post_hook("final_norm"))

        # The custom Gemma3 model has a different architecture from HF's Gemma,
        # so we can't load weights directly.
        # This test only checks if the model runs without error.
        model_config = ModelConfig(pretrained_config=hf_gemma_config,
                                   attn_backend=backend)
        gemma = Gemma3ForCausalLM(model_config).to(dtype).to(device).eval()
        gemma.load_weights(hf_gemma.state_dict())

        # Load tokenizer to convert prompt to real input_ids
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(GEMMA3_MODEL_PATH)

        # Test prompt - you can customize this
        test_prompt = "Hello world, this is a test prompt for Gemma 3 model evaluation."

        # Tokenize the prompt to get real input_ids
        input_ids = tokenizer.encode(test_prompt,
                                     return_tensors="pt",
                                     add_special_tokens=True)
        input_ids = input_ids.squeeze(0).to(
            device)  # Remove batch dimension and move to device

        print(f"Test prompt: '{test_prompt}'")
        print(f"Input tokens shape: {input_ids.shape}")
        print(f"Input tokens: {input_ids.tolist()}")
        # breakpoint()

        # context
        tokens_per_block = 32
        num_blocks = (input_ids.size(-1) // tokens_per_block) + 1
        head_dim = gemma.config.head_dim
        num_layers = gemma.config.num_hidden_layers
        num_kv_heads = gemma.config.num_key_value_heads
        max_seq_len = num_blocks * tokens_per_block
        batch_size = 1

        max_num_tokens = 8192

        if dtype == torch.half:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
        elif dtype == torch.bfloat16:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
        else:
            raise ValueError("Invalid dtype")

        mapping = Mapping(world_size=1, tp_size=1, rank=0)

        kv_cache_config = KvCacheConfig(max_tokens=num_blocks *
                                        tokens_per_block)
        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )

        num_cached_tokens_per_seq = [0]
        request_ids = [1]
        token_nums = [input_ids.size(-1)]
        prompt_lens = [input_ids.size(-1)]
        kv_cache_manager.add_dummy_requests(request_ids,
                                            token_nums,
                                            is_gen=False)

        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            ),
            max_num_requests=1,
            max_num_tokens=max_num_tokens,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
        )

        position_ids = [torch.arange(0, input_ids.size(-1))]
        position_ids = torch.cat(position_ids).unsqueeze(0).cuda()
        with torch.inference_mode():
            # Clear any previous intermediate tensors
            from tensorrt_llm._torch.models.modeling_gemma3 import (
                clear_intermediate_tensors, get_intermediate_tensors)
            clear_intermediate_tensors()

            if DEBUG_INTERMEDIATE_OUTPUTS:
                print("\n" + "=" * 50)
                print("RUNNING HF MODEL FORWARD PASS")
                print("=" * 50)

            # Run HF model first to capture intermediates
            ref = hf_gemma.forward(input_ids=input_ids.unsqueeze(0),
                                   position_ids=position_ids,
                                   use_cache=True)
            debug_log_hf_tensor("HF Final logits", ref.logits[:, -1])

            if DEBUG_INTERMEDIATE_OUTPUTS:
                print("\n" + "=" * 50)
                print("RUNNING TRT-LLM MODEL FORWARD PASS")
                print("=" * 50)

            # Run TRT-LLM model
            attn_metadata.prepare()
            logits = gemma.forward(input_ids=input_ids,
                                   position_ids=position_ids,
                                   attn_metadata=attn_metadata)

            # Get TRT-LLM intermediate tensors
            trtllm_intermediates = get_intermediate_tensors()

            if DEBUG_INTERMEDIATE_OUTPUTS:
                print("\n" + "=" * 50)
                print("COMPARING INTERMEDIATE OUTPUTS")
                print("=" * 50)

                # Compare embedding if available
                if "embedding" in trtllm_intermediates and "embedding" in hf_intermediates:
                    compare_tensors("Embedding",
                                    trtllm_intermediates["embedding"],
                                    hf_intermediates["embedding"])

                # Compare layer-wise intermediates (first few layers as example)
                num_layers_to_compare = min(3,
                                            hf_gemma_config.num_hidden_layers)
                for i in range(num_layers_to_compare):
                    # TRTLLM and HF share the same layer variable names
                    layer_comparisons = [
                        ("input_layernorm_pre",
                         f"layer_{i}_input_layernorm_pre"),
                        ("input_layernorm", f"layer_{i}_input_layernorm"),
                        ("self_attn_q_norm", f"layer_{i}_self_attn_q_norm"),
                        ("self_attn_k_norm", f"layer_{i}_self_attn_k_norm"),
                        ("self_attention", f"layer_{i}_self_attention"),
                        ("post_attention_layernorm",
                         f"layer_{i}_post_attention_layernorm"),
                        ("pre_feedforward_layernorm",
                         f"layer_{i}_pre_feedforward_layernorm"),
                        ("mlp", f"layer_{i}_mlp"),
                        ("post_feedforward_layernorm",
                         f"layer_{i}_post_feedforward_layernorm"),
                    ]

                    for component, key in layer_comparisons:
                        assert key in trtllm_intermediates and key in hf_intermediates
                        success = compare_tensors(f"Layer {i}: {key}",
                                                  trtllm_intermediates[key],
                                                  hf_intermediates[key],
                                                  layer_idx=i)
                        if not success:
                            # compare corresponding component weights
                            # hf_weights = hf_gemma.model.layers[i].state_dict()[f"{component}.weight"]
                            # trtllm_weights = gemma.model.layers[i].state_dict()[f"{component}.weight"]
                            breakpoint()

                print("=" * 50)

        print(f"logits.shape: {logits.shape}")
        torch.testing.assert_close(logits,
                                   ref.logits[:, -1].float(),
                                   atol=0.4,
                                   rtol=0.4)

        # gen
        # gen_input_ids = torch.tensor([[600]], dtype=torch.int, device=device)

        # num_cached_tokens_per_seq = torch.tensor([input_ids.size(-1)],
        #                                          device=device,
        #                                          dtype=torch.int)

        # attn_metadata.kv_cache_params.num_cached_tokens_per_seq = num_cached_tokens_per_seq
        # attn_metadata.is_context = False

        # gen_position_ids = torch.tensor(
        #     [[input_ids.size(-1)]], dtype=torch.int,
        #     device=device)  # position of the new token

        # def run_forward(input_ids, position_ids, attn_metadata):
        #     attn_metadata.prepare()
        #     if not scenario.use_cuda_graph:
        #         return gemma.forward(input_ids=input_ids,
        #                              position_ids=position_ids,
        #                              attn_metadata=attn_metadata)
        #     else:
        #         graph_runner = DecodingCUDAGraphRunner(
        #             attn_metadata.max_num_requests, "cuda", attn_metadata)
        #         graph_runner.capture(lambda inputs: gemma.forward(**inputs))

        #         for _ in range(2):
        #             attn_metadata.prepare()
        #             logits = graph_runner.run({
        #                 "input_ids": input_ids,
        #                 "position_ids": position_ids,
        #                 "attn_metadata": attn_metadata,
        #             })
        #         return logits

        # if scenario.use_cuda_graph:
        #     attn_metadata = attn_metadata.create_cuda_graph_metadata(1)

        # with torch.inference_mode():
        #     logits = run_forward(input_ids=gen_input_ids,
        #                          position_ids=gen_position_ids,
        #                          attn_metadata=attn_metadata)
        #     ref = hf_gemma.forward(input_ids=gen_input_ids,
        #                            position_ids=gen_position_ids,
        #                            past_key_values=ref.past_key_values,
        #                            use_cache=True)

        # self.assertIsNotNone(logits)
        # torch.testing.assert_close(logits,
        #                            ref.logits[:, -1],
        #                            atol=1e-2,
        #                            rtol=1e-2)

        # kv_cache_manager.free()
