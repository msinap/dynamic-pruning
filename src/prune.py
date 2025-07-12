from copy import deepcopy
import torch.nn as nn

# def prune_layers(model, layer_idxs, adapters=None):
#     """
#     Prune the layers of the model.
#     If adapters are provided, use them to replace the layers.
#     If adapters are not provided, use Identity() to replace the layers.
#     """
#     pruned_model = deepcopy(model)
#     for layer_idx in layer_idxs:
#         if adapters:
#             pruned_model.model.layers[layer_idx] = deepcopy(adapters[layer_idx])
#         else:
#             pruned_model.model.layers[layer_idx] = nn.Identity()
        
#     return pruned_model

class LLMPruner:
    """
    Prune the layers of the model.
    If adapters are provided, use them to replace the layers.
    If adapters are not provided, use Identity() to replace the layers.
    """

    def __init__(self, llm_model, adapters=None):
        self.llm_model_full = llm_model
        self.adapters = adapters
        self.original_layers_backup = {}

    def prune_model(self, layer_indices_to_replace_with_adapters):
        self._restore_original_layers() # Restore any previous state
        self.original_layers_backup.clear()

        for layer_idx in layer_indices_to_replace_with_adapters:
            self.original_layers_backup[layer_idx] = self.llm_model_full.model.layers[layer_idx]
            if self.adapters:
                self.llm_model_full.model.layers[layer_idx] = self.adapters[layer_idx]
            else:
                self.llm_model_full.model.layers[layer_idx] = nn.Identity()

    def _restore_original_layers(self):
        for layer_idx, original_layer in self.original_layers_backup.items():
            self.llm_model_full.model.layers[layer_idx] = original_layer
        self.original_layers_backup.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._restore_original_layers() # Ensure restoration on exit
