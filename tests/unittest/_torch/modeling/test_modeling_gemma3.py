import unittest
from dataclasses import dataclass

import torch
from parameterized import parameterized
from transformers import Gemma3ForCausalLM as HfGemmaForCausalLM
from transformers import Gemma3TextConfig
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
                    tensor1: torch.Tensor,
                    tensor2: torch.Tensor,
                    layer_idx: int = None):
    """Compare two tensors and log differences"""
    if not DEBUG_INTERMEDIATE_OUTPUTS:
        return

    layer_prefix = f"Layer-{layer_idx}: " if layer_idx is not None else ""
    print(f"[COMPARISON] {layer_prefix}{name}")

    # Ensure same shape
    if tensor1.shape != tensor2.shape:
        print(f"  Shape mismatch: {tensor1.shape} vs {tensor2.shape}")
        return

    # Calculate differences
    diff = (tensor1 - tensor2).float()
    abs_diff = torch.abs(diff)

    print(f"  Max absolute difference: {abs_diff.max().item():.6f}")
    print(f"  Mean absolute difference: {abs_diff.mean().item():.6f}")
    print(f"  Std of differences: {diff.std().item():.6f}")

    # Check if tensors are close
    try:
        torch.testing.assert_close(tensor1, tensor2, atol=1e-3, rtol=1e-3)
        print(f"  ✓ Tensors are close (atol=1e-3, rtol=1e-3)")
    except AssertionError:
        print(f"  ✗ Tensors differ significantly")

    print()


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
            enable_debug_logging(save_tensors=True)
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
        def create_hook(name):

            def hook(module, input, output):
                if DEBUG_INTERMEDIATE_OUTPUTS:
                    if isinstance(output, tuple):
                        # For attention layers that return (output, attention_weights, past_key_value)
                        tensor_output = output[0]
                    else:
                        tensor_output = output
                    hf_intermediates[name] = tensor_output.detach()
                    debug_log_hf_tensor(f"HF {name}", tensor_output)

            return hook

        # Register hooks for embedding and layers
        hf_gemma.model.embed_tokens.register_forward_hook(
            create_hook("embedding"))
        for i, layer in enumerate(hf_gemma.model.layers):
            layer.input_layernorm.register_forward_hook(
                create_hook(f"layer_{i}_input_layernorm"))
            layer.self_attn.register_forward_hook(
                create_hook(f"layer_{i}_self_attention"))
            layer.post_attention_layernorm.register_forward_hook(
                create_hook(f"layer_{i}_post_attention_layernorm"))
            layer.pre_feedforward_layernorm.register_forward_hook(
                create_hook(f"layer_{i}_pre_feedforward_layernorm"))
            layer.mlp.register_forward_hook(create_hook(f"layer_{i}_mlp"))
            layer.post_feedforward_layernorm.register_forward_hook(
                create_hook(f"layer_{i}_post_feedforward_layernorm"))
        hf_gemma.model.norm.register_forward_hook(create_hook("final_norm"))

        # The custom Gemma3 model has a different architecture from HF's Gemma,
        # so we can't load weights directly.
        # This test only checks if the model runs without error.
        model_config = ModelConfig(pretrained_config=hf_gemma_config,
                                   attn_backend=backend)
        gemma = Gemma3ForCausalLM(model_config).to(dtype).to(device).eval()
        gemma.load_weights(hf_gemma.state_dict())

        num_blocks = 1
        tokens_per_block = 128
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

        # context
        input_ids = torch.tensor([100, 200, 300, 100, 200, 100, 400, 500],
                                 dtype=torch.int,
                                 device=device)

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

            if DEBUG_INTERMEDIATE_OUTPUTS:
                print("\n" + "=" * 50)
                print("COMPARING INTERMEDIATE OUTPUTS")
                print("=" * 50)

                # Compare final logits first
                compare_tensors("Final logits", logits, ref.logits[:, -1])

                # Note: Due to architectural differences, we can't directly compare all intermediates
                # But we can see where they start diverging by looking at the debug logs above
                print(
                    "\nNote: Due to architectural differences between HF and TRT-LLM implementations,"
                )
                print(
                    "direct intermediate comparisons may not be possible for all layers."
                )
                print(
                    "Look at the debug logs above to identify where significant differences occur."
                )
                print("=" * 50)

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
