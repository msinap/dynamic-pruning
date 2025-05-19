import random
import torch
from datasets import Dataset as HuggingFaceDataset
from torch.utils.data import Dataset


def create_preference_dataset(samples_scenarios, ds):
    preference_dataset_raw = []
    for sample_id_key in samples_scenarios.keys():
        # Ensure sample_id_key is a valid index for ds['train']
        if sample_id_key >= len(ds['train']):
            print(f"Warning: sample_id {sample_id_key} from offline data is out of bounds for ds['train']. Skipping.")
            continue
        current_sample_data = ds['train'][sample_id_key]

        for _ in range(10): # Create 10 preference pairs per original prompt
            if len(samples_scenarios[sample_id_key]['scenarios']) < 2:
                continue
            scenario_w, scenario_l = random.sample(samples_scenarios[sample_id_key]['scenarios'], 2)

            # Determine winner/loser based on score, then number of pruned layers (fewer is better for tie)
            if scenario_w['score'] < scenario_l['score']:
                scenario_w, scenario_l = scenario_l, scenario_w
            elif scenario_w['score'] == scenario_l['score']:
                if len(scenario_w['pruned_layers']) > len(scenario_l['pruned_layers']): # Fewer pruned layers is preferred
                    scenario_w, scenario_l = scenario_l, scenario_w
                elif len(scenario_w['pruned_layers']) == len(scenario_l['pruned_layers']):
                    continue # Skip if identical after tie-breaking

            # Actor input text format
            actor_input_text = current_sample_data['tools'] + '\n' + current_sample_data['query']

            preference_dataset_raw.append({
                "input": actor_input_text,
                "pruned_layers_winner": scenario_w['pruned_layers'],
                "score_winner": scenario_w['score'],
                "pruned_layers_loser": scenario_l['pruned_layers'],
                "score_loser": scenario_l['score'],
            })
    random.shuffle(preference_dataset_raw)
    return preference_dataset_raw


# --- Preference Dataset and DataLoader ---
class PruningPreferenceDataset(Dataset):
    def __init__(self, data, tokenizer_actor, num_llm_layers_ds, max_seq_len):
        self.data = data
        self.tokenizer = tokenizer_actor
        self.num_llm_layers = num_llm_layers_ds
        self.max_seq_length = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokenized_input = self.tokenizer(
            item['input'], truncation=True, padding='max_length',
            max_length=self.max_seq_length, return_tensors="pt"
        )
        input_ids = tokenized_input.input_ids.squeeze(0)
        attention_mask = tokenized_input.attention_mask.squeeze(0)

        winner_k = len(item['pruned_layers_winner'])
        winner_ratio_val = float(winner_k) / self.num_llm_layers if self.num_llm_layers > 0 else 0.0
        loser_k = len(item['pruned_layers_loser'])
        loser_ratio_val = float(loser_k) / self.num_llm_layers if self.num_llm_layers > 0 else 0.0
        
        return {
            "input_ids": input_ids, "attention_mask": attention_mask,
            "winner_layers": torch.tensor(item['pruned_layers_winner'], dtype=torch.long),
            "winner_ratio": torch.tensor(winner_ratio_val, dtype=torch.float),
            "winner_k": torch.tensor(winner_k, dtype=torch.long),
            "loser_layers": torch.tensor(item['pruned_layers_loser'], dtype=torch.long),
            "loser_ratio": torch.tensor(loser_ratio_val, dtype=torch.float),
            "loser_k": torch.tensor(loser_k, dtype=torch.long),
        }

def collate_preference_data(batch):
    return {
        key: torch.stack([item[key] for item in batch])
        if key not in ["winner_layers", "loser_layers"] else [item[key] for item in batch]
        for key in batch[0].keys()
    }
