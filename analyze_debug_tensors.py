#!/usr/bin/env python3
"""
Utility script to analyze saved debug tensors from Gemma3 debugging
This script loads saved tensors and performs detailed comparisons
"""

import glob
import os
from typing import Dict, List, Tuple

import torch


def load_debug_tensors(
    debug_dir: str = "debug_tensors"
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Load HF and TRT-LLM debug tensors from files"""
    if not os.path.exists(debug_dir):
        print(f"Debug directory {debug_dir} not found!")
        print("Run the debug test with save_tensors=True first")
        return {}, {}

    hf_tensors = {}
    trtllm_tensors = {}

    # Load HF tensors
    hf_files = glob.glob(f"{debug_dir}/hf_*.pt")
    for file_path in hf_files:
        name = os.path.basename(file_path).replace("hf_", "").replace(".pt", "")
        hf_tensors[name] = torch.load(file_path)

    # Load TRT-LLM tensors
    trtllm_files = glob.glob(f"{debug_dir}/trtllm_*.pt")
    for file_path in trtllm_files:
        name = os.path.basename(file_path).replace("trtllm_",
                                                   "").replace(".pt", "")
        trtllm_tensors[name] = torch.load(file_path)

    print(
        f"Loaded {len(hf_tensors)} HF tensors and {len(trtllm_tensors)} TRT-LLM tensors"
    )
    return hf_tensors, trtllm_tensors


def analyze_tensor_pair(name: str, hf_tensor: torch.Tensor,
                        trtllm_tensor: torch.Tensor):
    """Analyze differences between a pair of tensors"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"{'='*60}")

    # Basic info
    print(f"HF Shape: {hf_tensor.shape}, TRT-LLM Shape: {trtllm_tensor.shape}")
    print(f"HF Dtype: {hf_tensor.dtype}, TRT-LLM Dtype: {trtllm_tensor.dtype}")

    # Handle shape differences (HF often has batch dimension)
    if hf_tensor.shape != trtllm_tensor.shape:
        if len(hf_tensor.shape) == len(
                trtllm_tensor.shape) + 1 and hf_tensor.shape[0] == 1:
            # Remove batch dimension from HF tensor
            hf_tensor = hf_tensor.squeeze(0)
            print(
                f"Removed batch dimension from HF tensor, new shape: {hf_tensor.shape}"
            )
        elif hf_tensor.shape != trtllm_tensor.shape:
            print("Shape mismatch - cannot compare directly")
            return

    # Convert to same dtype for comparison
    hf_float = hf_tensor.float()
    trtllm_float = trtllm_tensor.float()

    # Statistical comparison
    print(f"\nStatistical Comparison:")
    print(
        f"HF    - Mean: {hf_float.mean().item():.6f}, Std: {hf_float.std().item():.6f}"
    )
    print(
        f"TRT   - Mean: {trtllm_float.mean().item():.6f}, Std: {trtllm_float.std().item():.6f}"
    )
    print(
        f"HF    - Min: {hf_float.min().item():.6f}, Max: {hf_float.max().item():.6f}"
    )
    print(
        f"TRT   - Min: {trtllm_float.min().item():.6f}, Max: {trtllm_float.max().item():.6f}"
    )

    # Difference analysis
    diff = hf_float - trtllm_float
    abs_diff = torch.abs(diff)

    print(f"\nDifference Analysis:")
    print(f"Max absolute difference: {abs_diff.max().item():.6f}")
    print(f"Mean absolute difference: {abs_diff.mean().item():.6f}")
    print(f"Std of differences: {diff.std().item():.6f}")
    print(f"Median absolute difference: {abs_diff.median().item():.6f}")

    # Relative error
    rel_error = abs_diff / (torch.abs(hf_float) + 1e-8)
    print(f"Max relative error: {rel_error.max().item():.6f}")
    print(f"Mean relative error: {rel_error.mean().item():.6f}")

    # Closeness tests with different tolerances
    tolerances = [(1e-1, 1e-1), (1e-2, 1e-2), (1e-3, 1e-3), (1e-4, 1e-4)]
    print(f"\nCloseness Tests:")
    for atol, rtol in tolerances:
        try:
            torch.testing.assert_close(hf_float,
                                       trtllm_float,
                                       atol=atol,
                                       rtol=rtol)
            print(f"✓ Close with atol={atol}, rtol={rtol}")
        except AssertionError:
            print(f"✗ Not close with atol={atol}, rtol={rtol}")

    # Element-wise analysis for small tensors
    if hf_tensor.numel() <= 20:
        print(f"\nElement-wise comparison (tensor is small):")
        print(f"HF values:     {hf_float.flatten()}")
        print(f"TRT-LLM values: {trtllm_float.flatten()}")
        print(f"Differences:   {diff.flatten()}")


def create_tensor_name_mapping() -> Dict[str, str]:
    """Create mapping between HF and TRT-LLM tensor names"""
    mapping = {}

    # Basic mappings - based on actual saved tensor names
    mapping["HF_Final_logits"] = "Final_logits"
    mapping["HF_embedding"] = "After_embedding"
    mapping["HF_final_norm"] = "After_final_norm"

    # Layer-specific mappings (Gemma3 1B has 26 layers: 0-25)
    for i in range(32):  # Support up to 32 layers
        # Main layer comparisons
        mapping[
            f"HF_layer_{i}_output"] = f"After_MLP_residual_layer_{i}"  # Final output of layer
        mapping[
            f"HF_layer_{i}_attention"] = f"After_self_attn_layer_{i}"  # Attention output
        mapping[f"HF_layer_{i}_mlp"] = f"After_MLP_layer_{i}"  # MLP output

        # Additional TRT-LLM intermediate points that don't have HF equivalents
        # These are useful for debugging but won't have HF counterparts:
        # - Input_to_layer_layer_{i}
        # - After_input_layernorm_layer_{i}
        # - After_post_attention_layernorm_layer_{i}
        # - After_attention_residual_layer_{i}
        # - After_pre_feedforward_layernorm_layer_{i}
        # - After_post_feedforward_layernorm_layer_{i}

    return mapping


def find_matching_pairs(
        hf_tensors: Dict[str, torch.Tensor],
        trtllm_tensors: Dict[str, torch.Tensor]) -> List[Tuple[str, str, str]]:
    """Find matching tensor pairs between HF and TRT-LLM"""
    pairs = []

    # Get the mapping
    name_mapping = create_tensor_name_mapping()

    # Try direct matches first
    for hf_name in hf_tensors.keys():
        if hf_name in trtllm_tensors:
            pairs.append((hf_name, hf_name, hf_name))

    # Try mapped matches
    for hf_name, trtllm_name in name_mapping.items():
        if hf_name in hf_tensors and trtllm_name in trtllm_tensors:
            pairs.append((hf_name, trtllm_name, f"{hf_name} <-> {trtllm_name}"))

    # Pattern-based matching for more complex cases
    hf_names = set(hf_tensors.keys())
    trtllm_names = set(trtllm_tensors.keys())

    print(f"\nDebug - Available HF tensor names:")
    for name in sorted(hf_names):
        print(f"  '{name}'")

    print(f"\nDebug - Available TRT-LLM tensor names:")
    for name in sorted(trtllm_names):
        print(f"  '{name}'")

    print(f"\nDebug - Mapping attempts:")
    for hf_name, trtllm_name in name_mapping.items():
        hf_exists = hf_name in hf_tensors
        trtllm_exists = trtllm_name in trtllm_tensors
        print(
            f"  {hf_name} -> {trtllm_name}: HF={hf_exists}, TRT={trtllm_exists}"
        )

    # Remove duplicates
    seen = set()
    unique_pairs = []
    for hf_name, trtllm_name, display_name in pairs:
        key = (hf_name, trtllm_name)
        if key not in seen:
            seen.add(key)
            unique_pairs.append((hf_name, trtllm_name, display_name))

    return unique_pairs


def main():
    """Main analysis function"""
    print("Gemma3 Debug Tensor Analyzer")
    print("=" * 40)

    # Load tensors
    hf_tensors, trtllm_tensors = load_debug_tensors()

    if not hf_tensors and not trtllm_tensors:
        return

    print(f"\nAvailable HF tensors:")
    for name in sorted(hf_tensors.keys()):
        print(f"  - {name}: {hf_tensors[name].shape}")

    print(f"\nAvailable TRT-LLM tensors:")
    for name in sorted(trtllm_tensors.keys()):
        print(f"  - {name}: {trtllm_tensors[name].shape}")

    # Find matching pairs
    pairs = find_matching_pairs(hf_tensors, trtllm_tensors)

    if not pairs:
        print("\nNo matching tensor pairs found!")
        print("This might be due to different naming conventions.")
        return

    print(f"\nFound {len(pairs)} matching pairs:")
    for hf_name, trtllm_name, display_name in pairs:
        print(f"  - {display_name}")

    # Show TRT-LLM only tensors for internal analysis
    matched_trtllm = {trtllm_name for _, trtllm_name, _ in pairs}
    trtllm_only = set(trtllm_tensors.keys()) - matched_trtllm
    if trtllm_only:
        print(f"\nTRT-LLM only tensors (useful for internal debugging):")
        for name in sorted(trtllm_only):
            print(f"  - {name}")

    # Analyze each pair
    for hf_name, trtllm_name, display_name in pairs:
        analyze_tensor_pair(display_name, hf_tensors[hf_name],
                            trtllm_tensors[trtllm_name])

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(
        "Look for the first tensor pair where significant differences appear.")
    print(
        "\nFor additional debugging, you can also examine TRT-LLM internal tensors"
    )
    print("like normalization outputs and residual connections listed above.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
