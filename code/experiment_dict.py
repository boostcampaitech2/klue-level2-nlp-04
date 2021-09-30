def get_experiment_dict():
    experiment_list = {
        0: "", 1: "Model", 2: "Loss", 3: "Batch",
        4: "LR", 5: "Warmup-Steps", 6: "K-Fold",
        7: "Multi-Label",
    }
    model_list = {
        0: ("klue/roberta-base", "KLUE-RoBERTa-base"),
        1: ("klue/roberta-large", "KLUE-RoBERTa-large"),
        2: ("klue/roberta-small", "KLUE-RoBERTa-small"),
        3: ("klue/bert-base", "KLUE-BERT-base"),
        4: ("monologg/koelectra-base-v3-discriminator", "KOELECTRA-base")
    }

    return experiment_list, model_list
