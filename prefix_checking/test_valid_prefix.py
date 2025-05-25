#!/usr/bin/env python3
"""
Test script for the is_valid_output_prefix function.
"""

import sys
import os

# Add parent directory to path to import from src
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from src.llm import is_valid_output_prefix
except ImportError:
    try:
        from prefix_checking import is_valid_output_prefix
    except ImportError:
        # If running as __main__
        from . import is_valid_output_prefix

def test_valid_prefix():
    # Example tools
    tools = [
        {
            "name": "live_giveaways_by_type",
            "description": "Retrieve live giveaways from the GamerPower API based on the specified type.",
            "parameters": {
                "type": {
                    "description": "The type of giveaways to retrieve (e.g., game, loot, beta).",
                    "type": "str",
                    "default": "game",
                    "required": True
                }
            }
        },
        {
            "name": "search_games",
            "description": "Search for games by name.",
            "parameters": {
                "query": {
                    "description": "Search query",
                    "type": "str",
                    "required": True
                },
                "limit": {
                    "description": "Maximum number of results",
                    "type": "int",
                    "required": False
                }
            }
        }
    ]
    
    # Test cases: (prefix, expected_result, description)
    test_cases = [
        # Valid prefixes
        ("", True, "Empty prefix"),
        ("[", True, "Just opening bracket"),
        ("[{", True, "Array with object start"),
        ('[{"name":', True, "Object with name key"),
        ('[{"name": "live_giveaways_by_type"', True, "Valid function name partial"),
        ('[{"name": "live_giveaways_by_type"}', True, "Complete function name"),
        ('[{"name": "live_giveaways_by_type", "arguments": {', True, "Starting arguments"),
        ('[{"name": "live_giveaways_by_type", "arguments": {"type": "beta"}}]', True, "Complete valid call"),
        ('[{"name": "search_games", "arguments": {"query": "test"', True, "Partial arguments"),
        
        # Invalid prefixes
        ('[{"name": "invalid_function"', False, "Invalid function name"),
        ('[{"invalid_key":', False, "Invalid object key"),
        ('[{"name": "live_giveaways_by_type", "arguments": {"invalid_arg":', False, "Invalid argument name"),
        ('[[', False, "Double array opening"),
        ('[}', False, "Mismatched brackets"),
        ('[{"name": "live_giveaways_by_type"}}', False, "Extra closing brace"),
        
        # Partial strings that could be valid
        ('[{"name": "live_', True, "Partial function name that matches"),
        ('[{"name": "search_', True, "Partial function name for search_games"),
        ('[{"name": "live_giveaways_by_type", "arguments": {"ty', True, "Partial argument name"),
        
        # Multiple function calls
        ('[{"name": "live_giveaways_by_type", "arguments": {"type": "beta"}}, {"name": "search_games"', True, "Multiple calls partial"),
        
        # Edge cases
        ('[{"name": "live_giveaways_by_type", "arguments": {}}]', False, "Missing required argument"),
        ('[{"name": "search_games", "arguments": {"query": "test", "limit": 10}}]', True, "Optional argument included"),
    ]
    
    print("Testing is_valid_output_prefix function:\n")
    print("-" * 80)
    
    passed = 0
    failed = 0
    
    for prefix, expected, description in test_cases:
        result = is_valid_output_prefix(tools, prefix)
        status = "✓" if result == expected else "✗"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
            
        print(f"{status} {description}")
        print(f"  Prefix: {repr(prefix)}")
        print(f"  Expected: {expected}, Got: {result}")
        if result != expected:
            print(f"  ERROR: Result doesn't match expectation!")
        print()
    
    print("-" * 80)
    print(f"Summary: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    
    return failed == 0


if __name__ == "__main__":
    success = test_valid_prefix()
    sys.exit(0 if success else 1) 