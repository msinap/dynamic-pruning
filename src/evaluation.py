import json
from src.llm import generate_llm_output

def exact_match_score(output, answer):
    return 1 if output == answer else 0

def json_match_score(output, answer):
    try:
        return 1 if json.loads(output) == json.loads(answer) else 0
    except Exception:
        return 0

def ratio_function_calls_score(output, answer):
    try:
        output_json = json.loads(output)
        answer_json = json.loads(answer)
        match_count = 0
        for function_call in answer_json:
            if function_call in output_json:
                match_count += 1
        return match_count / len(answer_json)
    except Exception:
        return 0

def partial_match_score(output_str, answer_str):
    try:
        output_json = json.loads(output_str)
        parsed_output = {}
        if not isinstance(output_json, list): # Handle if output is not a list of calls
            return -5 # Malformed output
        for func_call in output_json:
            if not isinstance(func_call, dict) or 'name' not in func_call or 'arguments' not in func_call:
                return -5 # Malformed function call
            if not isinstance(func_call['arguments'], dict): # Arguments must be a dict
                 return -5
            parsed_output[func_call['name']] = func_call['arguments']
    except Exception:
        return -5 # Cannot parse output

    answer_json = json.loads(answer_str)
    if len(answer_json) == 0:
        return 10 if len(parsed_output) == 0 else 0

    score = 0.0
    for expected_func_call in answer_json:
        single_call_max_score = 10.0 / len(answer_json)
        if expected_func_call['name'] not in parsed_output: # Function name mismatch or missing
            continue # No points for this expected call

        score += single_call_max_score / 2.0 # Half points for correct function name
        
        # Argument matching
        expected_args = expected_func_call['arguments']
        predicted_args = parsed_output[expected_func_call['name']]
        
        if not expected_args: # If no arguments expected and name matches
            if not predicted_args: # and no arguments predicted
                 score += single_call_max_score / 2.0
            continue


        num_expected_args = len(expected_args)
        if num_expected_args == 0: continue # Already handled if name matched

        arg_score_per_item = (single_call_max_score / 2.0) / num_expected_args
        
        for arg_name, arg_value in expected_args.items():
            if arg_name not in predicted_args:
                # score -= arg_score_per_item / 3.0 # Penalty for missing argument (optional, can be harsh)
                pass
            elif predicted_args[arg_name] == arg_value:
                score += arg_score_per_item # Full point for this arg
            # else: # Argument value mismatch (optional: partial credit or penalty)
                # score += arg_score_per_item / 3.0 # Partial for key match, value mismatch (optional)

    return max(0, score) # Ensure score is not negative from penalties if any

def evaluate_model_on_dataset(model, tokenizer, dataset, score_funcs, verbose=False):
    total_scores = {score_func.__name__: 0 for score_func in score_funcs}
    for i, sample in enumerate(dataset):
        output = generate_llm_output(sample, model, tokenizer)
        if verbose:
            print(f"Sample {i}")
            print(f"Answer: {sample['answers']}")
            print(f"Output: {output}")
        for score_func in score_funcs:
            score = score_func(output, sample['answers'])
            total_scores[score_func.__name__] += score
            if verbose:
                print(f"Score {score_func.__name__}: {score}")
    average_scores = {k: v / len(dataset) for k, v in total_scores.items()}
    return average_scores
