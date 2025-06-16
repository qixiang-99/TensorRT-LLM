#!/usr/bin/env python3
"""
Debug script for Gemma3 TensorRT-LLM vs HuggingFace comparison
This script will run the test with detailed debug logging enabled.
"""

import os
import subprocess
import sys


def run_debug_test():
    """Run the Gemma3 test with debug logging enabled"""
    print("Running Gemma3 debug test...")
    print("This will show detailed intermediate outputs from both models")
    print(
        "Look for significant differences in the debug logs to identify divergence points"
    )
    print("=" * 80)

    # Change to the test directory
    test_file = "tests/unittest/_torch/modeling/test_modeling_gemma3.py"

    if not os.path.exists(test_file):
        print(f"Error: Test file {test_file} not found!")
        print(
            "Make sure you're running this from the TensorRT-LLM root directory"
        )
        return 1

    # Run the specific test
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_file + "::TestGemma3::test_gemma3_allclose_to_hf",
        "-v",
        "-s"  # verbose and don't capture output
    ]

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"Error running test: {e}")
        return 1


if __name__ == "__main__":
    print("""
Gemma3 Debug Test Runner
========================

This script will run the Gemma3 comparison test with detailed debug logging.
You'll see intermediate outputs from both the HuggingFace and TensorRT-LLM models.

Key things to look for in the output:
1. Compare the statistics (mean, std, min, max) at each layer
2. Look for the first point where significant differences appear
3. Pay attention to differences in:
   - After embedding
   - After each layer's attention
   - After each layer's MLP
   - After final norm
   - Final logits

The debug output format:
- [HF DEBUG] shows HuggingFace model outputs
- [TRT-LLM DEBUG] shows TensorRT-LLM model outputs
- [COMPARISON] shows direct comparisons where possible

Additional features:
- To save intermediate tensors to files, modify the test to call:
  enable_debug_logging(save_tensors=True)
- Tensors will be saved to debug_tensors/ directory
- Use saved tensors for detailed numerical analysis in Python

""")

    sys.exit(run_debug_test())
