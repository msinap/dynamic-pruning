import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import wandb
from tqdm import tqdm

from code.llm import tokenize_for_llm

class Adapter(nn.Module):
    def __init__(self, io_dim, bottleneck_dim):
        super(Adapter, self).__init__()
        self.layer1 = nn.Linear(io_dim, bottleneck_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(bottleneck_dim, io_dim)
        self._initialize_weights(io_dim)

    def _initialize_weights(self, io_dim):
        std = 1.0 / math.sqrt(io_dim)
        nn.init.normal_(self.layer1.weight, mean=0.0, std=std)
        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, x, **_): # Match signature of LLM layer forward
        h = self.layer1(x[0] if isinstance(x, tuple) else x) # Handle tuple input from LLM layers
        h = self.relu(h)
        y = self.layer2(h)
        output = (x[0] if isinstance(x, tuple) else x) + y
        # Return in the same format as the original layer (often a tuple)
        if isinstance(x, tuple):
            return (output,) + x[1:]
        return (output,)

def load_adapters(adapter_path_template, adapter_io_dim, adapter_bottleneck_dim, num_llm_layers, device):
    llm_adapters = [
        Adapter(adapter_io_dim, adapter_bottleneck_dim).to(device=device, dtype=torch.bfloat16)
        for _ in range(num_llm_layers)
    ]
    print(f"Attempting to load adapters from paths like: {adapter_path_template.format(i=0)}")
    for i, adapter_module in enumerate(llm_adapters):
        adapter_module.load_state_dict(torch.load(adapter_path_template.format(i=i), map_location=device))
    return nn.ModuleList(llm_adapters) # Important for them to be proper model submodules if needed later

def train_adapters(adapters, ds, num_layers, tokenizer, model, num_samples, device, lr):
    optimizers = [torch.optim.Adam(adapters[i].parameters(), lr=lr) for i in range(num_layers)]
    for sample_id in tqdm(range(num_samples)):
        sample = ds['train'][sample_id]
        layers_input_output = [[] for _ in range(num_layers)]

        # generate data
        # todo: generate data in batch => not much difference?
        inputs = tokenize_for_llm(sample, tokenizer, device)
        input_ids_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                num_return_sequences=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                # do_sample=False,
            )
        # generated_tokens = outputs.sequences[:, input_ids_len:]
        # output = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        # if output != sample['answers']:
        #     print(f"Sample {sample_id} failed")
        #     print(output)
        #     print(sample['answers'])
        #     continue

        for step, step_hidden_states in enumerate(outputs.hidden_states):
            for layer_idx in range(num_layers):
                layers_input_output[layer_idx].append((
                    step_hidden_states[layer_idx].squeeze(0),  # intput
                    step_hidden_states[layer_idx+1].squeeze(0),  # output
                ))

        # train
        layer_losses = {}
        for i in range(num_layers):
            input_batch = torch.cat([input_tensor for input_tensor, _ in layers_input_output[i]], dim=0)
            output_batch = torch.cat([output_tensor for _, output_tensor in layers_input_output[i]], dim=0)
            adapter_output = adapters[i](input_batch)[0]
            loss = F.mse_loss(adapter_output, output_batch)
            layer_losses[f"loss_layer_{i}"] = loss.item()
            optimizers[i].zero_grad()
            loss.backward()
            optimizers[i].step()
            
        wandb.log(layer_losses)


def save_adapters(adapters, run_id):
    os.makedirs(f'/workspace/adapter_models/1_{run_id}/', exist_ok=True)
    for i, adapter in enumerate(adapters):
        torch.save(adapter.state_dict(), f'/workspace/adapter_models/1_{run_id}/adapter_{i}.pth')
