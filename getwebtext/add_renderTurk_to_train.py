import pandas as pd
from renderTurk import append_MTurk_batch_response, majority_vote
from train_utils import read_config
from collections import defaultdict

if __name__ == "__main__":
    batch_file="./MTurkBatch/Batch_4766827_batch_results.csv"
    df_vote = append_MTurk_batch_response(batch_file)
    df_maj_vote = majority_vote(df_vote)
    print("MTurk voted on %d companies." % len(df_maj_vote))
    config = read_config("config/tte_small.yaml")
    print(df_maj_vote)
    keep_cols = [
        config["score_col"],
        config["query_col"],
        config["ref_col"]
    ]
    eval_company = pd.read_csv(config['MTurk_eval_csv'])[['company', 'tic']]
    df_maj_vote = pd.merge( df_maj_vote, eval_company, on='tic').drop_duplicates('tic')
    print(len(df_maj_vote), 'valid rows.')
    df_maj_vote.to_csv('MTurk/eval_MTurk_gt.csv', )
    df_extra = defaultdict(list)
    for _, row in df_maj_vote.iterrows():
        for phrase in set(df_maj_vote.columns)-{'tic', 'company'}:
            df_extra[config["query_col"]].append(phrase)
            df_extra[config["ref_col"]].append(row['company'])
            df_extra[config["score_col"]].append(row[phrase])

    df_extra = pd.DataFrame.from_dict(df_extra)
    print(df_extra)
    df_train = pd.read_csv(config['train_data_path'])
    df_train = df_train[keep_cols]
    df_train = df_train[df_train[config['ref_col']].isin(eval_company['company'])]
    print(df_train)
    df_train = pd.concat([df_extra , df_train] , axis=0, ignore_index=True)
    df_train.to_csv('data_model/tte_training_data_unnorm_wgt.csv', index=False)


