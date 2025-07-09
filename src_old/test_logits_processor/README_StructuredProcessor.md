# StructuredFunctionCallProcessor Testing and Demo

This directory contains test scripts and demonstrations for the `StructuredFunctionCallProcessor` class, which enforces structured function call generation during language model inference.

## Overview

The `StructuredFunctionCallProcessor` is a `LogitsProcessor` that constrains language model generation to produce valid JSON function calls based on a predefined set of tools/functions. It ensures:

- Valid JSON structure
- Function names match available tools
- Argument names match tool parameters
- Required arguments are included
- Proper nesting and syntax

## Files

### Core Implementation
- `src/llm.py` - Contains the `StructuredFunctionCallProcessor` class and related utilities

### Test Scripts
- `test_structured_processor.py` - Comprehensive test suite with multiple test scenarios
- `simple_test.py` - Simple, focused tests that don't require large models
- `demo_generation.py` - Demonstration of actual text generation with the processor

## Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers
```

### 2. Run Simple Tests

```bash
cd dynamic-pruning
python simple_test.py
```

This will test the core functionality without requiring large language models.

### 3. Run Full Test Suite

```bash
python test_structured_processor.py
```

This includes more comprehensive tests and edge cases.

### 4. Run Generation Demo

```bash
python demo_generation.py
```

This demonstrates actual text generation with the processor (requires downloading a small model like GPT-2).

## Usage Example

```python
from src.llm import StructuredFunctionCallProcessor, generate_structured_function_calls
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# Define your tools
tools = [
    {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
            "location": {"type": "string", "description": "City name"},
            "units": {"type": "string", "description": "Temperature units"}
        },
        "required": ["location"]
    }
]

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("your-model-name")
model = AutoModelForCausalLM.from_pretrained("your-model-name")

# Create sample
sample = {
    "query": "What's the weather in Paris?",
    "tools": json.dumps(tools)
}

# Generate structured function calls
result = generate_structured_function_calls(
    sample=sample,
    model_llm=model,
    tokenizer_llm=tokenizer,
    max_new_tokens=100,
    temperature=0.1
)

print(result)  # Should output valid JSON function calls
```

## Test Scenarios

### Simple Tests (`simple_test.py`)
- ✅ Processor initialization
- ✅ JSON state parsing at different stages
- ✅ Token validation for various states
- ✅ Logits processing with dummy inputs

### Comprehensive Tests (`test_structured_processor.py`)
- ✅ Full generation pipeline (with model loading)
- ✅ Edge cases and error handling
- ✅ Malformed tools handling
- ✅ Empty tools list handling
- ✅ Long function names

### Generation Demos (`demo_generation.py`)
- ✅ Real generation with small models
- ✅ Step-by-step manual generation
- ✅ State transition visualization
- ✅ Token validation during generation

## Key Features Tested

### 1. JSON Structure Validation
The processor ensures generated text follows valid JSON array structure:
```json
[
  {
    "name": "function_name",
    "arguments": {
      "param1": "value1",
      "param2": "value2"
    }
  }
]
```

### 2. Function Name Validation
Only function names that exist in the provided tools are allowed during generation.

### 3. Parameter Validation
- Parameter names must match those defined in the tool schema
- Required parameters are enforced
- Type hints are considered (string, number, boolean, array)

### 4. State Machine
The processor maintains a state machine that tracks:
- Current position in JSON structure
- Current function being generated
- Current arguments being filled
- Partial tokens and completions

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```
   Error: Could not load tokenizer/model
   ```
   - Ensure you have internet connection for downloading models
   - Try alternative models (gpt2, microsoft/DialoGPT-small)
   - Check available disk space

2. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'src.llm'
   ```
   - Ensure you're running from the `dynamic-pruning` directory
   - Check that `src/llm.py` exists

3. **Token Validation Issues**
   ```
   No valid tokens found for state
   ```
   - This is expected behavior when the processor is working correctly
   - It means invalid tokens are being blocked as intended

### Performance Notes

- The processor adds computational overhead during generation
- For production use, consider:
  - Using GPU acceleration
  - Optimizing token validation logic
  - Caching frequently used computations
  - Using larger, more capable models

## Expected Output

When running the tests, you should see output like:

```
=== Basic Functionality Test ===
✓ Processor initialized with 2 tools
✓ Valid function names: {'get_weather', 'calculate'}

=== JSON State Parsing Test ===
✓ '['... -> expecting: object_start
✓ '[{'... -> expecting: name_key
✓ '[{"name":'... -> expecting: name_value
...

=== Token Validation Test ===
✓ State 'list_start': 1 valid tokens
  Examples: '['
✓ State 'object_start': 2 valid tokens
  Examples: '{', ' '
...
```

## Contributing

When adding new tests:
1. Follow the existing test structure
2. Include both positive and negative test cases
3. Test edge cases and error conditions
4. Document expected behavior
5. Use descriptive test names and comments

## License

This code is part of the dynamic-pruning project and follows the same license terms. 