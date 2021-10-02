from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, Trainer, TrainingArguments
from experiment_dict import get_experiment_dict

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        model_path, config=config)


def ray_hp_space():
    from ray import tune
    return {
        "learning_rate": tune.loguniform(5e-6, 5e-4),
        "num_train_epochs": tune.choice(range(1, 6)),
        "seed": tune.choice(range(1, 42)),
    }


def ray(model_name):
    MODEL_NAME = model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)

    trainer = Trainer(
        model_init=model_init,  # NOTE: 반드시 model_init 함수로 모델을 불러와야합니다.
        args=training_args,
    )


def main():
    experiment_list, model_list = get_experiment_dict()
    model_name, wandb_name = model_list[0]
    experiment_name = experiment_list[1]

    ray(model_name=model_name,
        )

if __name__ == '__main__':
    main()