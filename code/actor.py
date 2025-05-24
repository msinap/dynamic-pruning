import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from huggingface_hub import HfApi
import os
import json

class Actor(nn.Module):
    def __init__(self, base_bert_model, head_hidden_dim, num_llm_layers_actor, log_std_min, log_std_max):
        super(Actor, self).__init__()
        self.config = deepcopy(base_bert_model.config)
        self.bert = deepcopy(base_bert_model) # base_bert_model should be the core BertModel

        self.classifier_hidden = nn.Linear(self.config.hidden_size, head_hidden_dim)
        self.classifier_activation = nn.ReLU()
        self.classifier_score = nn.Linear(head_hidden_dim, num_llm_layers_actor)
        self.classifier_ratio_mu = nn.Linear(head_hidden_dim, 1)
        self.classifier_ratio_log_std = nn.Linear(head_hidden_dim, 1)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        hidden = self.classifier_activation(self.classifier_hidden(pooled_output))
        layers_log_probs = F.log_softmax(self.classifier_score(hidden), dim=-1)
        mu_ratio = torch.sigmoid(self.classifier_ratio_mu(hidden))
        # Smooth approximation of clamp using sigmoid
        raw_log_std = self.classifier_ratio_log_std(hidden)
        log_std_ratio = self.log_std_min + (self.log_std_max - self.log_std_min) * torch.sigmoid(raw_log_std)
        return layers_log_probs, mu_ratio, log_std_ratio


def get_pruning_action_from_actor(
        actor_model, actor_input_text, tokenizer_actor,
        num_total_llm_layers, max_seq_len_actor, current_device
    ):
    actor_model.eval() # Ensure actor is in eval mode
    tokenized = tokenizer_actor(
        actor_input_text, truncation=True, padding='max_length',
        max_length=max_seq_len_actor, return_tensors="pt"
    )
    input_ids_actor = tokenized.input_ids.to(current_device)
    attention_mask_actor = tokenized.attention_mask.to(current_device)

    with torch.no_grad():
        layers_log_probs, mu_ratio, _ = actor_model(input_ids_actor, attention_mask_actor)

    mu_ratio_scalar = mu_ratio.squeeze().item()
    k_pruned = int(round(mu_ratio_scalar * num_total_llm_layers))
    k_pruned = max(0, min(k_pruned, num_total_llm_layers)) # Clamp k

    layer_scores = layers_log_probs.squeeze() # No need for exp if just taking topk
    
    if k_pruned > 0:
        _, top_k_indices = torch.topk(layer_scores, k_pruned)
        pruned_indices_list = top_k_indices.cpu().tolist()
    else:
        pruned_indices_list = []
        
    return pruned_indices_list, k_pruned, mu_ratio_scalar
    

def save_actor_to_hf_hub(actor_model, CONFIG):
    save_dir = f"/workspace/actor_model/{CONFIG['wandb_run_id']}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(actor_model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

    config_dict = {
        "head_hidden_dim": CONFIG["actor_head_hidden_dim"],
        "num_llm_layers": CONFIG["num_llm_layers"],
        "log_std_min": CONFIG["log_std_min"], 
        "log_std_max": CONFIG["log_std_max"],
        "base_bert_model": CONFIG["actor_base_model"]
    }
    
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config_dict, f)

    api = HfApi()
    api.upload_folder(
        folder_path=save_dir,
        repo_id=f"msinap/actor_model_{CONFIG['wandb_run_id']}",
        repo_type="model"
    )

    print(f"Model saved to: msinap/actor_model_{CONFIG['wandb_run_id']}")
