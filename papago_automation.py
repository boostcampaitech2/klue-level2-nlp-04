import sys
import time
import pandas as pd 
from selenium import webdriver
from tqdm import tqdm

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument("disable-gpu")

driver = webdriver.Chrome("C:\\chromedriver_win32\\chromedriver", options=chrome_options)

driver.maximize_window()

# 한국어 문장을 영어, 일본어, 중국어로 번역합니다.
def kor_to_trans(text_data, trans_lang): 
    """ trans_lang에 넣는 파라미터 값: 'en' -> 영어 'ja&hn=0' -> 일본어 'zh-CN' -> 중국어(간체) """ 
    trans_list = []
    for i in tqdm(range(len(text_data))): 
        try: 
            driver.get('https://papago.naver.com/?sk=ko&tk='+trans_lang+'&st='+text_data[i])
            time.sleep(2) 
            backtrans = driver.find_element_by_css_selector("div#txtTarget").text
            
        except: 
            try:
                driver.get('https://papago.naver.com/?sk=ko&tk='+trans_lang) 
                driver.find_element_by_css_selector("textarea#txtSource").send_keys(text_data[i])
                time.sleep(2) 
                backtrans = driver.find_element_by_css_selector("div#txtTarget").text
            except:
                backtrans = text_data[i]
        
        
        print(backtrans)
        if "XKZ" not in backtrans:
            backtrans = "XKZ" + backtrans
        if "KVX" not in backtrans:
            backtrans += "KVX"
        trans_list.append(backtrans)

    return trans_list


# 번역된 문장을 다시 한국어로 back translation 합니다
def trans_to_kor(text_data, transed_list, transed_lang):
    back_trans_list = []
    for i in tqdm(range(len(transed_list))): 
        try: 
            driver.get('https://papago.naver.com/?sk='+transed_lang+'&tk=ko&st='+transed_list[i])
            time.sleep(2.5) 
            backtrans = driver.find_element_by_css_selector("div#txtTarget").text


        except: 
            try:
                driver.get('https://papago.naver.com/?sk='+transed_lang+'&tk=ko') 
                driver.find_element_by_css_selector("textarea#txtSource").send_keys(transed_list[i])
                time.sleep(2.5) 
                backtrans = driver.find_element_by_css_selector("div#txtTarget").text
            except:
                # 에러가 발생하면 원래 한국어 문장을 리턴합니다
                backtrans = text_data[i]

        if "XKZ" not in backtrans:
            backtrans = "XKZ" + backtrans
        if "KVX" not in backtrans:
            backtrans += "KVX"
        back_trans_list.append(backtrans)
    return back_trans_list


# 고유명사를 XKZ와 KVX로 대치합니다 (heuristic하게 설정했습니다)
def proper_noun(df):
    for idx in range(len(df)):

        sub_entity = eval(df.iloc[idx, 2])
        sub_word = sub_entity['word']
        sub_s_idx = sub_entity['start_idx']
        sub_e_idx = sub_entity['end_idx']

        obj_entity = eval(df.iloc[idx, 3])
        obj_word = obj_entity['word']
        obj_s_idx = obj_entity['start_idx']
        obj_e_idx = obj_entity['end_idx']


        df.iloc[idx, 1] = df.iloc[idx, 1].replace(sub_word, "XKZ")
        df.iloc[idx, 1] = df.iloc[idx, 1].replace(obj_word, "KVX")


# XKZ와 KVX를 다시 원래 단어로 복원합니다
def recover_proper_noun(df):
    for idx in range(len(df)):
        sub_entity = eval(df.iloc[idx, 2])
        sub_word = sub_entity['word']
        
        obj_entity = eval(df.iloc[idx, 3])
        obj_word = obj_entity['word']
        
        df.iloc[idx, 1] = df.iloc[idx, 1].replace("XKZ", sub_word)
        df.iloc[idx, 1] = df.iloc[idx, 1].replace("KVX", obj_word)

        sub_s_idx = df.iloc[idx, 1].find(sub_word)
        sub_e_idx = sub_s_idx + len(sub_word) -1

        obj_s_idx = df.iloc[idx, 1].find(obj_word)
        obj_e_idx = obj_s_idx + len(obj_word) - 1

        df.iloc[idx, 2] = "{'word': '"+str(sub_word)+"', 'start_idx': "+str(sub_s_idx)+", 'end_idx': "+str(sub_e_idx)+"}"
        df.iloc[idx, 3] = "{'word': '"+str(obj_word)+"', 'start_idx': "+str(obj_s_idx)+", 'end_idx': "+str(obj_e_idx)+"}"



df = pd.read_csv("C:\\Users\\User\\Desktop\\papago\\train.csv")
# 특정 수 이하의 문장을 가지는 라벨을 augmentation하기 위해 선택합니다
aug_label = []
for label in df['label'].unique():
    if (df['label'].value_counts() > 400)[label] == True and (df['label'].value_counts() <= 450)[label] == True:
        aug_label.append(label)

corpus = []
for idx in range(len(df)):
    if df.loc[idx, 'label'] in aug_label:
        corpus.append(df.loc[idx])
corpus_df = pd.DataFrame(corpus)


proper_noun(corpus_df)

for la, lang in zip(['eng', 'jpn', 'ch'], ['en', 'ja&hn=0', 'zh-CN']):
    transed_list = []
    backtransed_list = []
    exec(f'df_{la} = corpus_df.copy()')
    transed_list.extend(kor_to_trans(corpus_df['sentence'].values, lang))
    print(f"{la}언어로 번역된 문장: {transed_list}")
    backtransed_list.extend(trans_to_kor(corpus_df['sentence'].values, transed_list, lang))
    exec(f'df_{la}["sentence"] = backtransed_list')
    print(f"{la}언어가 다시 한국어로 번역된 corpus: {backtransed_list}")


for df in [df_eng, df_jpn, df_ch]:
    recover_proper_noun(df)

df_integrated = pd.concat([df_eng, df_jpn, df_ch])

df_integrated.to_csv("C:\\Users\\User\\Desktop\\papago\\backtranslated_corpus_400_450.csv")