import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


def get_router_and_tokenizer(
    router_base_model_name: str,
    head_hidden_dim: int,
    num_llm_layers: int,
    log_std_min: float,
    log_std_max: float,
    device: torch.device,
):
    router_base_model = AutoModel.from_pretrained(router_base_model_name).to(device, dtype=torch.bfloat16)
    router_tokenizer = AutoTokenizer.from_pretrained(router_base_model_name)
    router = Router(
        base_bert_model=router_base_model, #.model,
        head_hidden_dim=head_hidden_dim,
        num_llm_layers=num_llm_layers,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
    ).to(device, dtype=torch.bfloat16)
    # for param in router.bert.parameters():
    #     param.requires_grad = False
    return router, router_tokenizer


class Router(nn.Module):
    """
    A router model that decides which layers of an LLM to prune dynamically.
    It outputs:
    1. Log probabilities for each layer's pruning score.
    2. Parameters (mu, log_std) for a Normal distribution over the pruning ratio.
    """
    def __init__(self, base_bert_model, head_hidden_dim, num_llm_layers, log_std_min, log_std_max):
        super(Router, self).__init__()
        self.config = deepcopy(base_bert_model.config)
        self.bert = deepcopy(base_bert_model)
        self.num_llm_layers = num_llm_layers

        self.layers_hidden = nn.Linear(self.config.hidden_size, head_hidden_dim)
        self.layers_activation = nn.ReLU()
        self.layers_score = nn.Linear(head_hidden_dim, self.num_llm_layers)
        
        self.ratio_hidden = nn.Linear(self.config.hidden_size, head_hidden_dim)
        self.ratio_activation = nn.ReLU()
        self.ratio_mu = nn.Linear(head_hidden_dim, 1)
        self.ratio_log_std = nn.Linear(head_hidden_dim, 1)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # mean pooling
        #pooled_output = outputs.last_hidden_state.mean(dim=1)
        token_embeddings = outputs[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        pooled_output = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = pooled_output.to(torch.bfloat16)
        
        hidden_layers = self.layers_activation(self.layers_hidden(pooled_output))
        layers_scores = F.sigmoid(self.layers_score(hidden_layers))
        
        # Parameters for the Normal distribution over the pruning ratio
        hidden_ratio = self.ratio_activation(self.ratio_hidden(pooled_output))
        mu_ratio = torch.sigmoid(self.ratio_mu(hidden_ratio))
        # Clamp the log_std to a reasonable range for stability
        raw_log_std = self.ratio_log_std(hidden_ratio)
        log_std_ratio = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (torch.tanh(raw_log_std) + 1)
        
        return layers_scores, mu_ratio, log_std_ratio



