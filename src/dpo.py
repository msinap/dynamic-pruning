

def train_dpo(
    router_model,
    router_tokenizer,
    llm_model,
    llm_tokenizer,
    llm_adapters,
    preference_dataset_train,
    preference_dataset_eval,
    num_llm_layers,
):
    for step, batch in enumerate(preference_dataset_train):
        print(batch)
        break
        