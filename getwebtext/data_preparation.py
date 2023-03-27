import pandas as pd
import spacy
import time

from svo.extractor import get_svo
from ticker_utils import ticker_finder
from keybert import KeyBERT
from string_utils import *

kw_model = KeyBERT()
nlp = spacy.load("en_core_web_sm")
ignore_words=["announces","announce"]


def skip_words(phrase):
    for w in ignore_words:
        if w in phrase:
            i = phrase.find(w)
            phrase = phrase[i + len(w) + 1:]
    return phrase


def svo_cleaner(s,ext_ent=[]):

    head=str2linkedlist(s)
    for name in ext_ent:
        head = llmatch(head, name)

    entities = [X.text for X in nlp(s).ents if X.label_ == "ORG"]
    for name in entities:
        head = llmatch(head, name)

    phrase=linkedlist2str(head)
    phrase=skip_words(phrase)
    subj, object=get_svo(phrase)

    return str(subj),str(object)


def ner_cleaner(s,ext_ent=[]):
    subj, doc = get_svo(s)
    if not doc:
        doc=nlp(s)
    entities =[X.text for X in doc.ents if X.label_=="ORG"]
    words=' '.join([token.text for token in doc])
    head= str2linkedlist(words)
    for ent in ext_ent:
        #print('ent - ',ent)
        head = llmatch(head, ent.split())
    if subj:
        head = llmatch(head, str(subj).split())
    for ent in entities:
        #print('ent - ',ent)
        head=llmatch(head,ent.split())
    phrase=linkedlist2str(head)
    phrase=' '.join([x.text for x in nlp(phrase) if x.pos_ == "VERB" or x.pos_ == "NOUN"])
    return str(subj),phrase


def keyphrase(s):
    rake.extract_keywords_from_text(s)
    phrases=rake.get_ranked_phrases()
    rake.get_word_degrees()
    return phrases[:2]

def splitstockstr(x):
    x=x.encode("ascii", "ignore").decode()
    x=re.sub('[^A-Z]', ' ',x).split()[0]
    return (x.split(',')[0] if x else '').split(';')[0]

def test(df):
    # for algo. & func test purpose
    i=1597
    title = get_title_name(df["Title_of_Press_Announce"][i])
    company = getcompanyname(df["Company"][i])
    print(company, '---', title)

    words = ' '.join([token.text for token in nlp('I am a test sample.')])
    ll = str2linkedlist(words)
    print(ll)
    print(linkedlist2str(ll))
    print(title)
    print(company)
    _,ner=svo_cleaner(title, [company])
    print(ner)
    phrase=keyphrase(ner)
    print(phrase)
    return

rejectWords=['fiscal','quarter']


def ner_preprocessor(df):
    start_time= time.time()
    print('subject object extraction')
    df['comp'] = df['Company'].apply(lambda x: [x])
    df = df.drop(columns=['comp']).reset_index(drop=True)
    subj= []
    obj = []
    for i in range(len(df)):
        s,o = svo_cleaner(
            df['Title'][i].lower(),
            [df['Company'][i].lower()]
        )
        subj.append(str(s))
        obj.append(str(o))
        if i % 1000 ==0:
            print(i,'/', len(df), int(time.time()-start_time), ' secs.')
    df['subject'] = subj
    df['object'] = obj
    print('done, ', int(start_time-time.time()), ' secs.')
    has_fin = df['Title'].str.contains("financial result")
    has_quart = df['Title'].str.contains("quarter")
    df = df[~has_fin & ~has_quart]
    print('done', int(time.time()-start_time), ' secs.')
    return df

"""
TODO: build your own preprocessor here
Input: Pandas dataframe of the extracted text 
    df = pd.read_csv("data_model/extracted_text2.txt", sep="\t")
Output: pandas dataframe with your preprocessed Text columns
    return df_preprocessed["Company","Title","your_preprocessed_text_col"]
"""

def col_filter(df):
    df['Company'] = df['Company'].apply(lambda x: getcompanyname(x))
    df['Title'] = df["Title_of_Press_Announce"].apply(lambda x: get_title_name(x))
    df = df[df['Title'].notna() & df['Company'].notna()]
    has_fin = df['Title'].str.lower().str.contains("financial result")
    has_quart = df['Title'].str.lower().str.contains("quarter")
    df = df[~has_fin & ~has_quart]
    df = df[['Company', 'Title', 'date', 'Body']]
    print('body split')
    start_time=time.time()
    df['Body'] =df['Body'].apply(lambda x: ';;;;'.join(x.split("About ")[0].split(';;;;')[1:]) )
    print('done.', int(time.time()-start_time),' secs.')
    return df


def get_ref_company_text(ref_tickers, scrapped_files):
    df_all = []
    for fpath in scrapped_files:
        print('+ searching scrapped data', fpath)
        df = pd.read_csv(fpath, sep="\t")
        print('%d rows.' % len(df))
        companies = df['Company'].apply(lambda x: getcompanyname(x))
        comp = sorted(set(companies))
        found = {}
        for c in comp:
            if str(ticker_finder(c, False)) in ref_tickers:
                found[c]=True
                print('find ', c)
            else:
                found[c]=False
        found_comp = companies.apply(lambda x: found[x])
        df_all.append(df[found_comp])
    df_all = pd.concat(df_all, ignore_index=True).drop_duplicates()
    print("Referenced %d companies, found %d companies, %d sample texts." %
          (len(ref_tickers), len(set(df_all['Company'])), len(df_all)) )

    return df_all

def raw_experiment_pipeline():
    df = pd.read_csv("data_model/extracted_text2.txt", sep="\t")
    '''
    df0 = pd.read_csv("extracted_text1.txt", sep="\t")
    df= df.append(df0, ignore_index=True).drop_duplicates().reset_index()
    df0 = pd.read_csv("extracted_text2.txt", sep="\t")
    df= df.append(df0, ignore_index=True).drop_duplicates().reset_index()
    '''
    # test(df)

    for process_fn in [col_filter]:
        df = process_fn(df)
    df.to_csv("data_model/ner_cleaned_title.csv",index=False)
    return


def acct_experiment_pipeline():
    scrapped_files = [
        "data_model/extracted_text0.txt",
        "data_model/extracted_text1.txt",
        "data_model/extracted_text2.txt"
    ]
    df_ref = pd.read_csv("professional/company_using_analytics.csv", encoding='unicode_escape')
    ref_tickers = df_ref['Ticker'].str.upper().str.replace(' ','').tolist()
    df = get_ref_company_text(ref_tickers, scrapped_files)
    for process_fn in [col_filter]:
        df = process_fn(df)
    df.to_csv("data_model/acct_ner_cleaned_title.csv", index=False)
    df= df['Company'].value_counts()
    df = df.to_frame(name='count').reset_index()
    df = df.rename({'index': 'Company'}, axis=1)
    print(df)
    df['tic']= df['Company'].apply(lambda x: ticker_finder(x))
    df.to_csv("professional/eval_tics.csv", index=False)
    return

if __name__=="__main__":
    acct_experiment_pipeline()

