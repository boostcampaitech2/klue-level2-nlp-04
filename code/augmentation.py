from koeda import AEDA
from koeda import RD # Random Deletion
from koeda import RI # Random Insertion
from koeda import SR # Synonym Replacement
from koeda import RS # Reandom Swap

import numpy as np
import pandas as pd
import signal
import time
from tqdm import tqdm
import argparse
from multiprocessing import Process, Queue
pd.options.display.max_colwidth = 300

def aug_AEDA(sentence, ratio, morpheme, amount):
    augmenter = AEDA(
              morpheme_analyzer=morpheme,  # Default = "Okt"
              punc_ratio=0.3,
              punctuations=None  # default = ('.', ',', '!', '?', ';', ':')
            )
    return augmenter(
        data=sentence,
        p=ratio,  # Default = 0.3
        repetition=amount
    )


def aug_RD(sentence, ratio, morpheme, amount):
    augmenter = RD(
        morpheme_analyzer=morpheme,
    )

    return augmenter(
        data=sentence,
        p=ratio,
        repetition=amount
    )


def aug_RI(sentence, ratio, morpheme, amount):
    augmenter = RI(
        morpheme_analyzer=morpheme,
        stopword=False,
    )

    return augmenter(
        data=sentence,
        p=ratio,
        repetition=amount
    )


def aug_SR(sentence, ratio, morpheme, amount):
    augmenter = SR(
        morpheme_analyzer=morpheme,
        stopword=False,
    )

    return augmenter(
        data=sentence,
        p=ratio,
        repetition=amount
    )

def aug_RS(sentence, ratio, morpheme, amount):
    augmenter = RS(
        morpheme_analyzer=morpheme,
    )

    return augmenter(
        data=sentence,
        p=ratio,
        repetition=amount
    )


def augmentation(data):
    sentence = data.sentence
    subj = eval(data.subject_entity)['word']
    obj = eval(data.object_entity)['word']

    if count_dict[data.label] < target:
        amount = (target - count_dict[data.label]) // count_dict[data.label]
        amount = amount // 4
        if not amount:
            return

        results = []
        results += aug_AEDA(sentence=sentence, ratio=ratio, morpheme=morpheme, amount=amount)
        results += aug_RD(sentence=sentence, ratio=ratio, morpheme=morpheme, amount=amount)
        results += aug_RI(sentence=sentence, ratio=ratio, morpheme=morpheme, amount=amount)
        results += aug_SR(sentence=sentence, ratio=ratio, morpheme=morpheme, amount=amount)
        results += aug_RS(sentence=sentence, ratio=ratio, morpheme=morpheme, amount=amount)

        for ret in results:
            if subj not in ret or obj not in ret:
                continue
            aug_df.append([data.id, ret, dict({'word': subj}), dict({'word': obj}), data.label, data.source])


'''https://growd.tistory.com/57'''
class TimeOutException(Exception):
    pass

def alarm_handler(signum, frame):
    raise TimeOutException()

def run(idx):
    df.iloc[idx:idx + interval].progress_apply(augmentation, axis=1)

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0, help='augmentation start point (default: 0)')
parser.add_argument('--end', type=int, default=300, help='augmentation end point (default: 30000)')
args = parser.parse_args()

df = pd.read_csv('../dataset/train/stratified_train.csv')
# df.iloc[0].to_csv('../dataset/train/aeda/aeda_kkma_0.3.csv', index=False)
new_df = df.iloc[:2]

count_dict = dict(df.label.value_counts())
mean = int(df.label.value_counts().mean()) * 2
# mean = 1000
maxi = sorted(df.label.value_counts())[-2]  # no_relation is max and take second maximum

morpheme_analyzer_list = ["Okt", "Kkma", "Komoran", "Mecab", "Hannanum"]
morpheme = morpheme_analyzer_list[1]  # Kkma
ratio = 0.7

tqdm.pandas()

opt = 0  # mean : 0, maxi : 1
opt_name = "maxi" if opt else "mean"
target = [mean, maxi][opt]  # mean or maxi
interval = 50  # mean = 50, maxi = 50
sec = 60  # mean = 100sec, maxi = 200sec

size = len(df)
threads = 1
# start = 0
# end = size // threads
start = args.start * 100  # start and end are in 0 to 300(=len(df))
end = min(args.end * 100, size)
skips = []
add_time = [0, -30, -15, 240]
for idx in range(start, end+(interval if end % interval else 0), interval):
    repeat = 0
    while True:
        if repeat == 4:
            skips += f"skip idx : {idx}~{idx+interval}\n"
            print(f"<<<<<<<<<<<<<< skip range : {idx}~{idx+interval} >>>>>>>>>>>>>>")
            break
        print(f'======= progress {idx}~{idx + interval}',
              f'(({100*(idx-start)/(end-start):.2f}%) {idx // interval + 1} /',
              f'{end // interval + (1 if end % interval else 0)}',
              f'({size // interval + (1 if end % interval else 0)}))=======',
              f'start:{start//100} | end:{end//100} | sec:{sec+add_time[repeat]} | repeat:{repeat} | interval:{interval} |',
              f'opt_name:{opt_name}',
              sep=' ')

        aug_df = []
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(sec+add_time[repeat])
        try:
            run(idx)
        except TimeOutException as e:
            print("<<<<<<<<<<<<<< alert timeout before retry right now >>>>>>>>>>>>>>")
            repeat += 1
            continue
        aug_df = pd.DataFrame(data=aug_df, columns=df.columns)
        new_df = pd.concat([new_df, aug_df])
        break

new_df = new_df.iloc[2:]
new_df.to_csv(f'../dataset/train/aeda/train_aeda_Kkma_0.3_{opt_name}_{start}to{end}.csv', index=False)
print(skips)
print("======= finished =======")