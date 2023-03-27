
import pandas as pd
import numpy as np
"""
df_val = pd.read_csv("data_model/cf_validation_data.csv")
df_train = pd.read_csv("data_model/cf_training_data.csv")

df_val = df_val[df_val['scores']>0]
df_train = df_train[df_train['scores']>0]
df = pd.concat([df_train, df_val], ignore_index=True)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.drop_duplicates(subset=['Company', "Text"], keep='first')
df.to_csv("data_model/cf_eval_data.csv", index=False)

df =pd.read_csv("data_model/cf_eval_data.csv")
print(df['Company'].value_counts())
"""

import torchtext as tt
filepath = "data_model/cf_training_data_part0.csv"
RAW = tt.data.RawField()
TEXT = tt.data.Field(sequential=True,
  init_token='',  # start of sequence
  eos_token='',   # end of sequence
  lower=True,
  tokenize=tt.data.utils.get_tokenizer("basic_english"),)
LABEL = tt.data.Field(sequential=False,
  use_vocab=False,
  unk_token=None,
  is_target=True)

pos = tt.data.TabularDataset(
path='data/pos/pos_wsj_train.tsv', format='tsv',
fields=[('text', tt.data.Field()),
        ('labels', tt.data.Field())])

sentiment = tt.data.TabularDataset(
    path='data/sentiment/train.json', format='json',
    fields={'sentence_tokenized': ('text', tt.data.Field(sequential=True)),
             'sentiment_gold': ('labels', tt.data.Field(sequential=False))})
