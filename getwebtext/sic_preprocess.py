import pandas as pd

df=pd.read_csv("ais/sic.csv")
df = df[df['ni'].notna()]
#df=df.sort_values(by=['fyear'])
df = df.drop_duplicates(["tic"],keep="last")

#df=pd.DataFrame.from_dict({'tic':tic,'sic':sic,'ni':ni})
df=df[['tic','sic','ni']]
dfa=pd.read_csv("data_model/company_actions.csv")
dfa=dfa.merge(df,how='left',on='tic').reset_index()
dfa.to_csv("data_model/company_actions_acct.csv", index=False)