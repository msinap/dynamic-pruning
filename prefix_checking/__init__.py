"""
Prefix Checking Package

A package for validating function call prefixes in LLM outputs.
Includes utilities for filtering datasets and validating JSON prefixes.
"""

import sys
import os

# Add necessary directories to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
dynamic_pruning_dir = os.path.dirname(current_dir)  # This is now the parent directory
src_dir = os.path.join(dynamic_pruning_dir, 'src')

# Add dynamic-pruning to path so we can import from src
if dynamic_pruning_dir not in sys.path:
    sys.path.insert(0, dynamic_pruning_dir)

# Try to import from src/llm.py
is_valid_output_prefix = None

try:
    from src.llm import is_valid_output_prefix
except ImportError as e:
    # If that fails, try direct import
    try:
        from llm import is_valid_output_prefix
    except ImportError:
        pass

if is_valid_output_prefix is None:
    raise ImportError(
        f"Could not import is_valid_output_prefix function. "
        f"Looking for src/llm.py in: {src_dir}"
    )

__version__ = "1.0.0"
__all__ = ["is_valid_output_prefix"] 