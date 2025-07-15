import torch
import random
import numpy as np
import wandb

from src.llm import load_llm, load_and_split_dataset
from src.evaluation import evaluate_model_on_dataset, exact_match_score, json_match_score, ratio_function_calls_score
from src.adapter import train_adapters, Adapter, save_adapters

CONFIG = {
    "llm_model_name": "Salesforce/xLAM-2-1b-fc-r",
    "dataset_name": "Salesforce/xlam-function-calling-60k",
    "dataset_test_size": 33,
    "lr": 1e-4,
    "adapter_bottleneck_dim": 64,
    "num_samples_to_train": 5000,
    "eval_every_n_steps": 500,
    "wandb_id": f"{random.randint(1000, 9999)}",
}

seed = 23
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    wandb.init(
        project="adapter-training",
        name=f"adapter_training_{CONFIG['wandb_id']}",
        id=CONFIG["wandb_id"],
        config=CONFIG,
    )

    llm_model, llm_tokenizer = load_llm(CONFIG)
    test_dataset, train_dataset = load_and_split_dataset(
        CONFIG["dataset_name"],
        test_size=CONFIG["dataset_test_size"],
    )

    scores = evaluate_model_on_dataset(
        model=llm_model,
        tokenizer=llm_tokenizer,
        dataset=test_dataset,
        score_funcs=[exact_match_score, json_match_score, ratio_function_calls_score],
        verbose=False,
    )
    print(f"Scores: {scores}") # exact_match: 0.69, json_match: 0.71, ratio_function_calls: 0.78
    # Scores: {'exact_match_score': 0.5757575757575758, 'json_match_score': 0.6363636363636364, 'ratio_function_calls_score': 0.7272727272727273}

    adapters = [Adapter(CONFIG["llm_hidden_dim"], CONFIG["adapter_bottleneck_dim"]).to(device="cuda", dtype=torch.bfloat16) for _ in range(CONFIG["llm_num_layers"])]
    train_adapters(
        adapters=adapters,
        train_dataset=train_dataset.to_list()[:CONFIG["num_samples_to_train"]],
        num_layers=CONFIG["llm_num_layers"],
        tokenizer=llm_tokenizer,
        model=llm_model,
        lr=CONFIG["lr"],
        eval_every_n_steps=CONFIG["eval_every_n_steps"],
        eval_dataset=test_dataset,
        verbose=False,
    )
    # https://wandb.ai/sina-team/adapter-training/runs/1951

    save_adapters(
        adapters=adapters,
        run_id=CONFIG["wandb_id"],
        path_template="/workspace/models/adapter/{run_id}",
    )


