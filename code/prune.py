from copy import deepcopy


class LLMPruner:
    def __init__(self, model_llm, model_adapters):
        self.llm_model_full = model_llm
        self.adapters_list = model_adapters
        self.original_layers_backup = {}

    def prune_model(self, layer_indices_to_replace_with_adapters):
        self._restore_original_layers() # Restore any previous state
        self.original_layers_backup.clear()

        for layer_idx in layer_indices_to_replace_with_adapters:
            if 0 <= layer_idx < len(self.llm_model_full.model.layers):
                self.original_layers_backup[layer_idx] = self.llm_model_full.model.layers[layer_idx]
                self.llm_model_full.model.layers[layer_idx] = self.adapters_list[layer_idx]
            else:
                print(f"Warning: Invalid layer index {layer_idx} for pruning.")

    def _restore_original_layers(self):
        for layer_idx, original_layer in self.original_layers_backup.items():
            if 0 <= layer_idx < len(self.llm_model_full.model.layers):
                self.llm_model_full.model.layers[layer_idx] = original_layer
        self.original_layers_backup.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._restore_original_layers() # Ensure restoration on exit


def prune_layers(model, layer_idxs, adapters):
    pruned_model = deepcopy(model)    
    for layer_idx in layer_idxs:
        #pruned_model.model.layers[layer_idx] = Identity()
        pruned_model.model.layers[layer_idx] = deepcopy(adapters[layer_idx])
    return pruned_model
