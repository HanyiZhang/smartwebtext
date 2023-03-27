import pandas as pd

df = pd.read_csv("data_model/tte_training_data.csv")
nsamples = len(df)
query_col= 'Text'
for word in ['analytics', 'technology', 'innovation']:
    hasword = df[query_col].apply(lambda x: word in str(x))
    print('%d/%d query "%s" has %s.' % (hasword.sum(), nsamples, query_col, word))