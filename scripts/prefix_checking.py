from src.llm import load_llm, load_and_split_dataset
from src.evaluation import evaluate_model_on_dataset, exact_match_score, json_match_score

import torch
import random
import numpy as np

CONFIG = {
    "llm_model_name": "Salesforce/xLAM-2-1b-fc-r",
    "dataset_name": "Salesforce/xlam-function-calling-60k",
    "dataset_test_size": 100,
}

seed = 23
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    llm_model, llm_tokenizer = load_llm(CONFIG)
    test_dataset, train_dataset = load_and_split_dataset(CONFIG["dataset_name"], test_size=CONFIG["dataset_test_size"])

    scores = evaluate_model_on_dataset(llm_model, llm_tokenizer, test_dataset, [exact_match_score, json_match_score], verbose=True)
    print(f"Scores: {scores}") # exact_match: 0.69, json_match: 0.71

    



