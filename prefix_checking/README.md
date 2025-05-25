# Prefix Checking Package

A Python package for validating function call prefixes in LLM outputs. This package helps ensure that Language Model outputs for function calling tasks are syntactically valid at every prefix, making them suitable for training and evaluation.

## Features

- **Prefix Validation**: Validate that any prefix of a function call output is valid JSON and conforms to the available tools
- **Dataset Filtering**: Filter datasets to remove samples with invalid prefixes
- **Comprehensive Testing**: Test suites for validating individual prefixes and entire datasets
- **High Performance**: Processes ~50,000+ prefixes per second

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd prefix_checking

# Install the package
pip install -e .
```

## Quick Start

### Basic Usage

```python
from prefix_checking import is_valid_output_prefix

# Define available tools
tools = [
    {
        "name": "get_weather",
        "parameters": {
            "location": {"type": "str", "required": True},
            "unit": {"type": "str", "required": False}
        }
    }
]

# Validate prefixes
prefix = '[{"name": "get_weather", "arguments": {"location": "NYC"'
is_valid = is_valid_output_prefix(tools, prefix)
print(f"Is valid: {is_valid}")  # True
```

### Filtering a Dataset

```python
from prefix_checking.filter_dataset import has_all_valid_prefixes

# Check if all prefixes of an answer are valid
tools = [...]  # Your tools
answer = '[{"name": "function", "arguments": {...}}]'
all_valid = has_all_valid_prefixes(tools, answer)
```

## Command Line Tools

The package provides several command-line utilities:

### Filter Dataset
```bash
python -m prefix_checking.filter_dataset
```
Filters the Salesforce/xlam-function-calling-60k dataset to remove samples with invalid prefixes.

### Verify Dataset
```bash
python -m prefix_checking.verify_filtered_dataset
```
Verifies that a filtered dataset contains only valid samples.

### Test Prefixes
```bash
python -m prefix_checking.test_dataset_prefixes
```
Tests prefix validation on a subset of the dataset.

## Package Structure

```
prefix_checking/
├── __init__.py              # Package initialization and imports
├── filter_dataset.py        # Dataset filtering functionality
├── verify_filtered_dataset.py # Dataset verification tools
├── test_dataset_prefixes.py # Dataset testing utilities
├── test_full_dataset.py     # Full dataset validation
├── test_valid_prefix.py     # Unit tests for prefix validation
├── example_usage.py         # Usage examples
└── use_filtered_dataset.py  # Examples of using filtered datasets
```

## Validation Rules

The validator ensures:

1. **Valid JSON Structure**: Proper brackets, braces, quotes, and commas
2. **Function Name Validation**: Function names must match available tools
3. **Argument Validation**: 
   - Argument names must match tool parameters
   - Required arguments must be present in completed calls
   - Basic type checking for strings, numbers, booleans
4. **Nested Structure Support**: Handles arrays, objects, and nested combinations
5. **Partial String Matching**: Validates incomplete strings that could match valid completions

## Performance

- **Accuracy**: 98.33% on the Salesforce/xlam-function-calling-60k dataset
- **Speed**: ~54,000 prefixes validated per second
- **Dataset Filtering**: Retained 94.44% of samples (56,665 out of 60,000)

## Examples

See `example_usage.py` for comprehensive examples including:
- Basic validation
- Dataset filtering
- Working with filtered datasets
- Converting to different formats

## Testing

Run the test suite:
```bash
python -m prefix_checking.test_valid_prefix
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 