import torch
import torch.nn as nn
import math


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

def load_adapters(adapter_path_template, adapter_io_dim, adapter_bottleneck_dim, num_llm_layers_explicit, device):
    llm_adapters = [
        Adapter(adapter_io_dim, adapter_bottleneck_dim).to(device=device, dtype=torch.bfloat16)
        for _ in range(num_llm_layers_explicit)
    ]
    print(f"Attempting to load adapters from paths like: {adapter_path_template.format(i=0)}")
    for i, adapter_module in enumerate(llm_adapters):
        adapter_module.load_state_dict(torch.load(adapter_path_template.format(i=i), map_location=device))
    return nn.ModuleList(llm_adapters) # Important for them to be proper model submodules if needed later

