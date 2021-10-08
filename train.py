import argparse
import wandb
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, \
    TrainingArguments, EarlyStoppingCallback
from load_data import *
from utils import *
from model import CustomModel

# wandb description silent
os.environ['WANDB_SILENT'] = "true"


def train(train_df, valid_df, train_label, valid_label, args):
    # huggingface 에 저장된 model_name 또는 저장된 model_path
    MODEL_NAME = args.model_name
    # tokniezr load
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
    # typed_entity_marker 적용시 special_token 추가
    if args.tem:
        special_tokens_dict = {'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>',
                                                             '<e3>', '</e3>', '<e4>', '</e4>']}
        tokenizer.add_special_tokens(special_tokens_dict)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_df, tokenizer, args)
    tokenized_valid = tokenized_dataset(valid_df, tokenizer, args)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_valid_dataset = RE_Dataset(tokenized_valid, valid_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    # model_config load
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    if args.tem:
        # typed_entity_marker 적용시 Customized model load
        model = CustomModel(model_config, MODEL_NAME)
        model.model.resize_token_embeddings(len(tokenizer))
    else:
        # typed_entity_marker 미적용시 huggingface AutoModelForSequenceClassification model load
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    # print(model.config)
    # model 에 GPU 또는 CPU 에서 사용하기
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    # increment_path 사용해서 동일한 이름의 폴더명일 경우 이름뒤에 2, 3, ... 식으로 번호 추가
    output_dir = increment_path(os.path.join(args.output_dir, args.run_name))
    # training argument setting
    training_args = TrainingArguments(
        output_dir=output_dir,  # output directory
        save_total_limit=args.save_total_limit,  # number of total save model.
        save_steps=args.save_steps,  # model saving step.
        num_train_epochs=args.epochs,  # total number of training epochs
        learning_rate=args.learning_rate,  # learning_rate
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.valid_batch_size,  # batch size for evaluation
        warmup_steps=args.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,  # strength of weight decay
        logging_dir=args.logging_dir,  # directory for storing logs
        logging_steps=args.logging_steps,  # log saving step.
        evaluation_strategy=args.evaluation_strategy,  # evaluation strategy to adopt during training
        eval_steps=args.eval_steps,  # evaluation step.
        metric_for_best_model=args.metric_for_best_model,
        load_best_model_at_end=args.load_best_model_at_end,
        report_to=args.report_to,  # 'all', 'wandb', 'tensorboard'
    )
    # Trainer setting
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_valid_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],  # early stopping
    )

    # train model
    trainer.train()
    save_dir = increment_path(os.path.join('./best_model', args.model_name.split('/')[-1], args.run_name))
    model.save_pretrained(save_dir)

    # 마지막 최종 best model 로 평가한 결과 저장해서 return
    eval_result = trainer.evaluate(RE_valid_dataset)

    return eval_result


def main(args):
    # seed 고정해서 실험의 재현성 확보
    seed_everything(args.seed)

    # 본인의 datafile 을 넣어주세요
    train_dataset = load_data("../dataset/train/train.csv", args)

    # fold 별 f1_score 저장 리스트
    fold_valid_f1_list = []
    # StratifiedKFold 사용해서 전체 데이터를 5등분
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    # train_dataset['label'] 을 기준으로 데이터를 5등분
    for fold, (train_idx, valid_idx) in enumerate(skf.split(train_dataset, train_dataset['label']), 1):
        # cv=True 인 경우 5폴드를 모두 실행
        if not args.cv:
            # cv=False 인 경우 1폴드까지만 실행
            if fold > 1:
                break
        print(f'>> Cross Validation {fold} Starts!')

        # load dataset setting
        train_df = train_dataset.iloc[train_idx]
        valid_df = train_dataset.iloc[valid_idx]

        # train, valid label setting
        train_label = label_to_num(train_df['label'].values)
        valid_label = label_to_num(valid_df['label'].values)

        # wandb setting
        wandb_config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'model_name': args.model_name,
        }

        wandb.init(project=args.project_name,
                   name=f'{args.run_name}_{fold}',
                   config=wandb_config,
                   reinit=True,
                   )

        # train() 함수 실행하고 best model 에서 나온 f1_score 저장
        result = train(train_df, valid_df, train_label, valid_label, args)
        wandb.join()
        # fold 별 f1_score 저장 리스트에 저장
        fold_valid_f1_list.append(result['eval_micro f1 score'])

    # 최종 cross_validatoin 결과 출력
    print(f'cv_f1_score: {fold_valid_f1_list}')
    print(f'cv_f1_score: {np.mean(fold_valid_f1_list)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training arguments
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=10, help='total number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=35,
                        help='batch size per device during training (default: 35)')
    parser.add_argument('--valid_batch_size', type=int, default=128,
                        help='batch size for evaluation (default: 128)')
    parser.add_argument('--model_name', type=str, default='klue/roberta-large',
                        help='what kinds of models (default: klue/roberta-large)')
    parser.add_argument('--run_name', type=str, default='exp', help='name of the W&B run (default: exp)')
    parser.add_argument('--cv', type=bool, default=False, help='using cross validation (default: False)')
    parser.add_argument('--tem', type=bool, default=False, help='using typed entity marker (default: False)')

    # training arguments that don't change well
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='output directory (default: ./results)')
    parser.add_argument('--save_total_limit', type=int, default=2, help='number of total save model (default: 1)')
    parser.add_argument('--save_steps', type=int, default=100, help='model saving step (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='learning_rate (default: 5e-5)')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='number of warmup steps for learning rate scheduler (default: 300)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='strength of weight decay (default: 0.01)')
    parser.add_argument('--logging_dir', type=str, default='./logs',
                        help='directory for storing logs (default: ./logs)')
    parser.add_argument('--logging_steps', type=int, default=100,
                        help='log saving step (default: 100)')
    parser.add_argument('--evaluation_strategy', type=str, default='steps',
                        help='evaluation strategy to adopt during training (default: steps)')
    # `no`: No evaluation during training.
    # `steps`: Evaluate every `eval_steps`.
    # `epoch`: Evaluate every end of epoch.
    parser.add_argument('--eval_steps', type=int, default=100, help='evaluation step (default: 100)')
    parser.add_argument('--metric_for_best_model', type=str, default='micro f1 score',
                        help='metric_for_best_model (default: micro f1 score), log_loss')
    parser.add_argument('--load_best_model_at_end', type=bool, default=True, help='(default: True)')
    parser.add_argument('--report_to', type=str, default='wandb', help='(default: wandb)')
    parser.add_argument('--project_name', type=str, default='p_stage_klue',
                        help='wandb project name (default: p_stage_klue')
    parser.add_argument('--early_stopping_patience', type=int, default=7,
                        help='number of early_stopping_patience (default: 7)')

    args = parser.parse_args()
    print(args)

    main(args)
