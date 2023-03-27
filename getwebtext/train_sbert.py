import pandas as pd
import os
from train_utils import read_config
from train_utils import load as cf_load
from data_utils import get_context_csv_data_loader, get_sbert_csv_evaluator_dataset
from cf import train_phrase_features
from cf import SBertContextModel as cf_model
import csv
from string_utils import find_links
from ticker_utils import ticker_finder
from sentence_transformers import InputExample
from torch.utils.data import DataLoader


def main():
    config = read_config("config/sbert.yaml")
    model = cf_model(config)
    print("model initialized. " )
    if not config.get('skip_training', False):
        print("create training -validation dataloader")
        train_ds = []
        num_rows = sum(1 for _ in open(config["train_data_path"], 'r'))
        with open(config["train_data_path"], 'r') as fIn:
            reader = csv.DictReader(fIn, delimiter=',')
            for i, row in enumerate(reader):
                score = float(row[config['score_col']])
                texts= [row[config['ref_col']]+config['special_sep']+row[config['context_col']], row[config['query_col']]]
                train_ds.append( InputExample(texts=texts, label=score))
                if i%(10**3)==0:
                    print('%d/%d rows.' % (i, num_rows))
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=config['batch_size'])
        del train_ds
        val_ds = get_sbert_csv_evaluator_dataset(
            config["val_data_path"],
            context_col= config['context_col'],
            ref_col=config['ref_col'],
            query_col=config['query_col'],
            score_col=config['score_col'],
            special_sep=config['special_sep'],
            limit=config.get('limit', None)
        )
        logger_dir = config.get("logger_dir", "./lightning_logs")
        os.makedirs(logger_dir, exist_ok=True)
        print(" start training ...")
        model.train(train_dl, val_ds)

    print("load eval reference dataset")
    eval_company = pd.read_csv(config['MTurk_eval_csv'])['company']
    df_eval = pd.read_csv(config["eval_data_path"])
    df_eval = df_eval[df_eval[config['ref_col']].isin(eval_company)]
    print("MTurk eval on %d companies." % len(set(df_eval[config['ref_col']])) )
    print(df_eval[config['ref_col']].value_counts())
    MTurk_eval_path = "MTurk_cf_eval_data.csv"
    df_eval.to_csv(MTurk_eval_path, index=False)
    ref_dl = get_context_csv_data_loader(
        MTurk_eval_path, train_phrase_features,
        batch_size=config['batch_size'],
        clear_cache=config['clear_cache'],
        shuffle=False, sep=','
    )
    model.model = cf_load(model.model, config['saved_model_path'])
    print("Saved model %s Loaded." % config['saved_model_path'])
    print("test search for several keywords.")
    kws = ['analytic', 'innovation', 'technology']
    df_kw_pred = model.predict(kws, ref_dl)
    df_kw_pred['link'] = find_links(df_kw_pred["Company"], 'data_model/default_search_full.txt')
    df_kw_pred['tic'] = df_kw_pred['Company'].apply(lambda x: ticker_finder(x))
    df_kw_pred.to_csv("cf_predictions.csv",index=False)
    return


if __name__=="__main__":
    main()
