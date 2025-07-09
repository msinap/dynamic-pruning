import json
from src.llm import generate_llm_output

def exact_match_score(output, answer):
    return 1 if output == answer else 0

def json_match_score(output, answer):
    try:
        return 1 if json.loads(output) == json.loads(answer) else 0
    except json.JSONDecodeError:
        return 0

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
