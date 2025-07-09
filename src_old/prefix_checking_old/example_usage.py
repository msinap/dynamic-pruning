#!/usr/bin/env python3
"""
Example usage of the is_valid_output_prefix function.
"""

import sys
sys.path.append('src')
from llm import is_valid_output_prefix


def main():
    # Define available tools
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather information for a location",
            "parameters": {
                "location": {
                    "description": "The city and state, e.g. San Francisco, CA",
                    "type": "str",
                    "required": True
                },
                "unit": {
                    "description": "Temperature unit (celsius or fahrenheit)",
                    "type": "str",
                    "required": False,
                    "default": "celsius"
                }
            }
        },
        {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "query": {
                    "description": "Search query",
                    "type": "str",
                    "required": True
                }
            }
        }
    ]
    
    print("Function Call Prefix Validator Example")
    print("=" * 50)
    print("\nAvailable tools:")
    for tool in tools:
        print(f"- {tool['name']}: {tool['description']}")
    print("\n" + "=" * 50 + "\n")
    
    # Test various prefixes
    test_prefixes = [
        # Valid prefixes
        '[{"name": "get_weather", "arguments": {"location": "San Francisco, CA"}}]',
        '[{"name": "get_weather", "arguments": {"location": "NYC"',
        '[{"name": "search_',
        '[{"name": "get_weather", "arguments": {',
        '[{',
        '[',
        '',
        
        # Invalid prefixes
        '[{"name": "invalid_function"}]',
        '[{"name": "get_weather", "arguments": {"invalid_param": "value"}}]',
        '[{"name": "get_weather", "arguments": {}}]',  # Missing required param
        '[[',  # Nested arrays not allowed
        '[}',  # Mismatched brackets
    ]
    
    for prefix in test_prefixes:
        is_valid = is_valid_output_prefix(tools, prefix)
        status = "✓ VALID" if is_valid else "✗ INVALID"
        
        # Truncate long prefixes for display
        display_prefix = prefix if len(prefix) <= 60 else prefix[:57] + "..."
        
        print(f"{status}: {repr(display_prefix)}")
        
        # Add explanation for some cases
        if prefix == '[{"name": "get_weather", "arguments": {}}]' and not is_valid:
            print("         ^ Missing required parameter 'location'")
        elif prefix == '[{"name": "invalid_function"}]' and not is_valid:
            print("         ^ Function 'invalid_function' not in available tools")
        elif prefix == '[[' and not is_valid:
            print("         ^ Nested arrays are not allowed in function calls")
            
        print()


if __name__ == "__main__":
    main() 