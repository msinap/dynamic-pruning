import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


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
    