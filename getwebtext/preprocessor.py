import pandas as pd
from string_utils import stopword_pattern, fasttext_toolkits, rake, wnl, stemmer, punct_exclist, cachedStopwords, ignore_unicode_char
import time
from nltk.tokenize import sent_tokenize
import numpy as np
import re
from collections import defaultdict
import random
"""
TODO: add your own preprocessing function
Input: raw dataframe: df (dtype: pd.DataFrame)
Output: preprocessed dataframe: df (dtype: pd.DataFrame)
"""

# ------------------------------------ Split --------------------------------
def drop_dup_titles(df):
    df = df.drop_duplicates(subset='Title', keep="first")
    return df

def _lem_stem(s, return_list=False):
    list_of_str = s.split() if isinstance(s, str) else s
    s = [wnl.lemmatize(w) for w in list_of_str]
    s = [stemmer.stem(w) for w in s]
    if return_list: return s
    else: return ' '.join(s)


def _extract_keyphrases(s):
    rake.extract_keywords_from_text(s)
    return rake.get_ranked_phrases()


def clean_col_text(col, stopwords = cachedStopwords,
                   remove_stopwords=True, keyphrase_only=True,
                   lem_stem=True, punct=True, split_sent=False,
                   ft_home="./fasttext", lowercase=True):
    stopword_ptn = stopword_pattern(stopwords)
    #punct_trans_table_ = str.maketrans('', '', punct_exclist)
    punct_trans_table_ = str.maketrans(punct_exclist, ' ' * len(punct_exclist))
    ft_tool = fasttext_toolkits(ft_home)

    def _punct(s):
        s = s.translate(punct_trans_table_)
        return s

    def _keep_only_keyphrases(s):
        rake.extract_keywords_from_text(s)
        return '  '.join(rake.get_ranked_phrases())

    def _remove_stopwords(text):
        text = stopword_ptn.sub(' ', text)
        return text

    def _clean_col_fn(df):
        df = df.drop_duplicates(subset=col, keep="first")
        df = df[(df[col].str.lower()!='none') \
                & (df[col].str.lower()!='nan') \
                & (df[col]!='') & (df[col].notna())]
        print('processing column ',col)
        isen = ft_tool.detEN(df[col].tolist())
        print('det lang')
        start_time = time.time()
        df = df[isen]
        print(int(time.time() - start_time), ' secs.')
        print('tokenize sentence')
        df[col] = df[col].apply(lambda x: x.replace(';;;;', '. '))
        df[col] = df[col].apply(lambda x: sent_tokenize(str(x)))
        print(int(time.time() - start_time), ' secs.')
        if lem_stem:
            print('lemmatization, stem')
            df[col] = df[col].apply(lambda s: [_lem_stem(x) for x in s])
            print(int(time.time() - start_time), ' secs.')
        if punct:
            print('punctuation')
            df[col] = df[col].apply(lambda s: [_punct(x) for x in s])
            print(int(time.time() - start_time), ' secs.')
        if remove_stopwords:
            print('remove stopwords')
            df[col] = df[col].apply(lambda s: [_remove_stopwords(x) for x in s])
            df[col] = df[col].apply(lambda s: [re.sub(' +', ' ',x) for x in s])
            print(int(time.time() - start_time), ' secs.')
        if keyphrase_only:
            print('keep only keyphrase')
            df[col] = df[col].apply(lambda s: [_keep_only_keyphrases(x) for x in s])
            print(int(time.time() - start_time), ' secs.')
        print('explode split sentence')
        if not split_sent:
            df[col]= df[col].apply(lambda x: ' '.join(x))
        else:
            df = df.explode(col)
        df[col] = df[col].str.strip()
        df.replace('', np.nan, inplace=True)
        df = df.dropna()
        df= df.drop_duplicates(subset=[col])
        df = df.reset_index(drop=True)
        if lowercase:
            df[col] = df[col].apply(lambda x: str(x).lower())
        print(df[col])
        return df

    return _clean_col_fn


def merge_cols(cols, output_col="Text", explode=False):
    def _merge(df):
        df[output_col] = ''
        for col in cols:
            df[output_col] += ' ' + df[col]
        df[output_col]= df[output_col].apply(lambda x: x.strip())
        df = df.drop(cols, axis=1)
        print(df)
        return df

    def _merge_explode(df):
        df.replace('', np.nan, inplace=True)
        df = df.dropna()
        df_cols = [df[col].copy() for col in cols]
        df = df.drop(cols, axis=1)
        df_output = []
        for df_col in df_cols:
            df_tmp = df.copy()
            df_tmp[output_col] = df_col
            df_tmp = df_tmp.drop_duplicates(subset=[output_col], keep='first')
            df_output.append(df_tmp)
            #df[output_col] += ';;;;' + df[col].apply(lambda x: str(x).lower().rstrip())
        df_output = pd.concat(df_output, ignore_index=True)
        df_output = df_output.drop_duplicates(subset=[output_col], keep='first')
        df_output = df_output.reset_index(drop=True)
        print(df_output)
        return df_output
    if not explode:
        return _merge
    else:
        return _merge_explode


def explode_keyphrase(col, to_words=False):
    def _explode(df):
        print('explode keyphrase')
        start_time= time.time()
        df["phrase"] = df[col].apply(lambda s: _extract_keyphrases(s))
        df = df.explode("phrase")
        df = df.dropna()
        df["phrase"] = df["phrase"].str.strip()
        if to_words:
            df["phrase"] = df["phrase"].apply(lambda s: s.split())
            df = df.explode("phrase")
            df = df[df['phrase'].str.len()>3]
            df = df.dropna()
        df = df.drop_duplicates()
        print(int(time.time() - start_time), ' secs.')
        df = df.reset_index(drop=True)
        print(df)
        return df

    return _explode


def cond_freq_score(phrase_col, cond_col, vocab_path, normalize=True):
    def _phrase_freq(df):
        start_time= time.time()
        print("Assign score based on phrase frequency")
        if normalize:
            print(" -  lem stem normalize text")
            df['normalized_text']=df[phrase_col].apply(lambda x: _lem_stem( str(x)))
        print("%d secs" % int(time.time()-start_time))
        #df_temp=df[[cond_col, 'temp']]
        ref_col = 'normalized_text' if normalize else phrase_col
        scores = np.zeros(len(df))
        ref_words= defaultdict(set)
        vocab = set()
        for v in sorted(set(df[cond_col])):
            idx = df[cond_col]==v
            cnt = df[idx][ref_col].value_counts()
            total = sum(cnt)
            cnt = cnt/total
            scores[idx]= cnt[df[idx][ref_col]].values
            print(v, "{:d} secs.".format(int(time.time()-start_time)))
            ref_words[v] = set(cnt.index) # use lem_stemed words
            #ref_words[v] = set(df[idx][phrase_col]) # use raw key phrases
            vocab.update(ref_words[v])
        df['scores'] = scores
        print(df)
        print("write vocab to file %s." % vocab_path)
        with open(vocab_path, "w", encoding='utf-8') as f:
            for w in sorted(vocab):
                print(w, file=f)
        print("CREATED {:d} words. {:d} secs.".format(len(vocab), int(time.time() - start_time)))
        #df_neg = pd.concat([df,df], ignore_index=True)
        #df_neg = pd.concat([df], ignore_index=True)
        df_neg = df.copy()
        if normalize:
            df_neg = df_neg.drop(['normalized_text'], axis=1)
        df_neg['scores']= 0.0
        neg_words = df_neg[phrase_col].copy()
        for v in sorted(set(df_neg[cond_col])):
            idx = df_neg[cond_col] == v
            n_ids =  sum(idx)
            not_in_ref_words= vocab - ref_words[v]
            if n_ids<=len(not_in_ref_words):
                sampled_neg_words = random.sample(sorted(not_in_ref_words), n_ids)
            else:
                sampled_neg_words = random.choices( sorted(not_in_ref_words),  k=n_ids)
            neg_words[idx] = sampled_neg_words
            print(v, "negatives {:d} secs.".format(int(time.time() - start_time)))
        df_neg[phrase_col] = neg_words
        if normalize:
            df[phrase_col] = df['normalized_text'] # used lem_stemed phrase
            df = df.drop(['normalized_text'], axis=1)
        print(df_neg)
        return pd.concat([df, df_neg], ignore_index=True)
    return _phrase_freq


def drop_col_rows_has_key_terms(col, key_terms):
    def _drop(df):
        print("drop column rows with key terms for %s" % col)
        start_time = time.time()
        for term in key_terms:
            df = df.reset_index(drop=True)
            contains = df[col].apply(lambda x: term.lower() in str(x).lower())
            df = df[~contains]
        print(df[col])
        print("done. %d secs." % int(time.time() - start_time))
        return df
    return _drop


def clean_char():
    def _char_check(df):
        print("character cleaning")
        start_time = time.time()
        df = df.applymap(lambda x: ignore_unicode_char(str(x).lower()))
        df.replace('', np.nan, inplace=True)
        df = df.dropna()
        df.replace(',', ' ', inplace=True)
        print(df)
        print("done. %d secs." % int(time.time()-start_time))
        return df
    return _char_check


def run_pipeline(df, preproc_fns):
    for fn in preproc_fns:
        print(fn)
        df = fn(df)
    df = df.reset_index(drop=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df


def load_cached_preprocessed_csv(df=None, data_path='data_model/cached_training_data.csv'):
    df = pd.read_csv(data_path)
    return df


def pipeline(df, load=False):
    if load:
        return [load_cached_preprocessed_csv]

    company_names = list(set(df['Company'].str.lower().tolist()))
    company_abbv = [w.split()[0] for w in company_names]
    stopwords = cachedStopwords + company_names + company_abbv
    use_cols = ["Title", "Body"]
    preprocessings = [
        drop_dup_titles,
        clean_col_text('Title', stopwords=stopwords, keyphrase_only=False),
        clean_col_text('Body', stopwords=stopwords, keyphrase_only=True),
        merge_cols(use_cols)
    ]
    return preprocessings

"""
TODO: add your own preprocessng pipline below
"""

def cf_pipeline(df, load_path="", vocab_path="data_model/cf_vocab.txt"):
    def load_cached_data(df=None):
        return load_cached_preprocessed_csv(df,load_path)

    if load_path:
        return [load_cached_data]

    company_names = list(set(df['Company'].str.lower().tolist()))
    company_abbv = [w.split()[0] for w in company_names]
    stopwords = cachedStopwords + company_names + company_abbv
    use_cols = ["Title", "Body"]
    preprocessings = [
        drop_dup_titles,
        clean_col_text('Title', stopwords=stopwords, punct=True, remove_stopwords=True,
                       keyphrase_only=False, lem_stem=False, split_sent=True),
        clean_col_text('Body', stopwords=stopwords, punct=True, remove_stopwords=False,
                       keyphrase_only=False, lem_stem=False, split_sent=True),
        merge_cols(use_cols, output_col="Text", explode=True),
        explode_keyphrase("Text", to_words=True),
        cond_freq_score(
            phrase_col="phrase", cond_col="Company", vocab_path=vocab_path, normalize=False),
        clean_char()
    ]
    return preprocessings


def tte_pipeline(df, load_path="", vocab_path="data_model/cf_vocab.txt"):
    def load_cached_data(df=None):
        return load_cached_preprocessed_csv(df,load_path)

    if load_path:
        return [load_cached_data]

    company_names = list(set(df['Company'].str.lower().tolist()))
    company_abbv = [w.split()[0] for w in company_names]
    stopwords = company_names + company_abbv
    use_cols = ["Title", "Body"]
    preprocessings = [
        drop_dup_titles,
        clean_col_text('Title', stopwords=stopwords, punct=True, remove_stopwords=True,
                       keyphrase_only=False, lem_stem=False, split_sent=True),
        drop_col_rows_has_key_terms(
            col='Title',
            key_terms=['conference', 'report', 'quarter', 'market', 'finan', 'meeting', 'earning']),
        clean_col_text('Body', stopwords=stopwords, punct=True, remove_stopwords=True,
                       keyphrase_only=False, lem_stem=False, split_sent=True),
        drop_col_rows_has_key_terms(
            col='Body',
            key_terms=['facebook:', 'twitter:', 'for more information', 'click here']),
        merge_cols(use_cols, output_col="Text", explode=True),
        #explode_keyphrase("Text", to_words=True),
        cond_freq_score(
            phrase_col="Text", cond_col="Company", vocab_path=vocab_path, normalize=False),
        clean_char()
    ]
    return preprocessings