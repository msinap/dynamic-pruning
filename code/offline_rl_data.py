
def process_scenarios(offline_dataset_list):
    samples_scenarios = {}
    for scenario in offline_dataset_list:
        sample_id = scenario['id']
        if sample_id not in samples_scenarios:
            samples_scenarios[sample_id] = {
                "scenarios": [], "max_score": -float('inf'), "max_pruned_layers": -1,
            }
        samples_scenarios[sample_id]["scenarios"].append(scenario)
        if scenario['score'] > samples_scenarios[sample_id]['max_score']: # Use > for score
            samples_scenarios[sample_id]['max_score'] = scenario['score']
            # Tie-breaking: if scores are equal, prefer fewer pruned layers
            if len(scenario['pruned_layers']) > samples_scenarios[sample_id]['max_pruned_layers']:
                 samples_scenarios[sample_id]['max_pruned_layers'] = len(scenario['pruned_layers'])
        elif scenario['score'] == samples_scenarios[sample_id]['max_score']:
            if len(scenario['pruned_layers']) < samples_scenarios[sample_id]['max_pruned_layers']: # Prefer fewer layers for same score
                 samples_scenarios[sample_id]['max_pruned_layers'] = len(scenario['pruned_layers'])
    return samples_scenarios

