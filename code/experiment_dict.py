def get_experiment_dict():
    experiment_list = {
        0: "", 1: "Model", 2: "DataAug-AEDA",
        3: "DataAug-Pororo", 4: "Batch",
        5: "LR", 6: "Warmup_Steps", 7: "K-Fold"
    }
    model_list = {
        0: ("klue/roberta-base", "KLUE-RoBERTa-base"),
        1: ("klue/roberta-large", "KLUE-RoBERTa-large"),
        2: ("klue/roberta-small", "KLUE-RoBERTa-small"),
        3: ("klue/bert-base", "KLUE-BERT-base"),
        4: ("monologg/koelectra-base-v3-discriminator", "KOELECTRA-base"),
        5: ("xlm-roberta-base", "XLM-RoBERTa-base"),
        6: ("distilbert-base-uncased", "DistilBERT-base"),
        7: ("bert-base-multilingual-cased", "mBERT-base"),
    }

    return experiment_list, model_list
