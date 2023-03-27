import random
import os
import sys
import gc

import pandas as pd
import numpy as np
from preprocessor import cf_pipeline as preproc_pipeline
from preprocessor import run_pipeline as preprocess
from preprocessor import load_cached_preprocessed_csv
from train_utils import training_pipeline, read_config, save_best
from train_utils import load as cf_load
from data_utils import get_context_csv_data_loader, randsample_df_by_group_rate
from cf import train_phrase_features, prediction_pipeline
from cf import TTEContextModel as cf_model


def main():
    config = read_config("config/cf_small.yaml")
    #config = read_config(sys.argv[1])
    keep_cols = [
        config["score_col"],
        config["query_col"],
        config["context_col"],
        config["ref_col"]
    ]

    if not config.get("skip_prep_data", False):
        print('preprocess dataset for training')
        df = pd.read_csv(config['raw_data_path'])
        preprocessing_fns = preproc_pipeline(df, load_path="", vocab_path=config['vocab_path'])
        df = preprocess(df, preprocessing_fns)
        df.to_csv(config['preproc_data_path'])
        df = load_cached_preprocessed_csv(data_path=config['preproc_data_path'])
        df= df.reset_index(drop=True)
        df['positives'] = df[config["score_col"]]>0
        df_val = randsample_df_by_group_rate(df, ['Company',"positives"], rate=0.002)
        df_val = df_val[~df_val.index.duplicated(keep='first')]
        df = df[ ~df.index.isin(df_val.index)]
        df= df[keep_cols]
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.to_csv(config["train_data_path"])
        df_val = df_val[keep_cols]
        df_val.to_csv(config["val_data_path"])
        df_val_eval = df_val[df_val[config["score_col"]] > 0]
        df_train_eval = df[df[config["score_col"]] > 0]
        df_eval = pd.concat([df_train_eval, df_val_eval], ignore_index=True)
        df_eval = df_eval.loc[:, ~df_eval.columns.str.contains('^Unnamed')]
        df_eval = df_eval.drop_duplicates(subset=[config['ref_col'], config['context_col']], keep='first')
        df_eval.to_csv(config['eval_data_path'])
        #df_train = pd.read_csv("data_model/cf_training_data.csv")
    val_data_path = config["val_data_path"]
    train_data_path = config["train_data_path"]
    if config.get('eval_query_only', False):
        val_data_path = "data_model/cf_small_val_data.csv"
        train_data_path = "data_model/cf_small_train_data.csv"
        eval_company = pd.read_csv(config['MTurk_eval_csv'])['company']
        df_val = pd.read_csv(config['val_data_path'])
        df_val = df_val[df_val[config['ref_col']].isin(eval_company)]
        df_val.to_csv(val_data_path, index=False)
        print("Shrinked MTurk validation on %d companies." % len(set(df_val[config['ref_col']])))
        print(df_val[config['ref_col']].value_counts())
        df_train = pd.read_csv(config['train_data_path'])
        df_train = df_train[df_train[config['ref_col']].isin(eval_company)]
        df_train.to_csv(train_data_path, index=False)
        print("Shrinked MTurk training on %d companies." % len(set(df_train[config['ref_col']])))
        print(df_train[config['ref_col']].value_counts())
        del df_train
        del df_val
        gc.collect()

    part_files = [config["train_data_path"]]
    if config.get('suhffle_parts', False):
        print('sharding data into parts.')
        part_files = []
        nparts = config['num_parts']
        df = pd.read_csv(train_data_path)
        df = df.sample(frac=1).reset_index(drop=True)
        parts = np.array_split(np.arange(len(df)), nparts)
        shuffled_train_parts = parts.copy()
        random.shuffle(shuffled_train_parts)
        for i in range(len(shuffled_train_parts)):
            part_file = config['train_data_parts'].replace("*", str(i))
            df_parts = df[shuffled_train_parts[i][0]:shuffled_train_parts[i][-1] + 1]
            df_parts = df_parts.loc[:, ~df_parts.columns.str.contains('^Unnamed')]
            df_parts.to_csv(part_file)
            part_files.append(part_file)
            print("part %d saved." % i)
        print("done.")

    BEST_LOSS = np.Inf
    model = cf_model(config)
    print("model initialized. " )
    if not config.get('skip_training', False):
        print("create training -validation dataloader")
        logger_dir = config.get("logger_dir", "./lightning_logs")
        os.makedirs(logger_dir, exist_ok=True)
        print(" start training ...")
        print(" %d train parts found." % len(part_files))
        for e in range(config['epochs']):
            for train_data_part in part_files:
                train_dl = get_context_csv_data_loader(
                    train_data_part,
                    train_phrase_features,
                    batch_size=config['batch_size'],
                    clear_cache=config['clear_cache'],
                    shuffle=True,
                    sep=',',
                    max_line=10 ** 7,
                    limit=config.get('limit', None))
                model, _ = training_pipeline(
                    model,
                    train_dl,
                    None,
                    nepochs=1,
                    resume_ckpt= True,
                    monitor=config['monitor'])
                del train_dl
                gc.collect()

            print("Parts training Epoch %d completed." % e)
            val_dl = get_context_csv_data_loader(
                val_data_path, train_phrase_features,
                batch_size=config['batch_size'],
                clear_cache=config['clear_cache'],
                shuffle=False,
                sep=',',
                max_line=10 ** 7,
                limit=config.get('limit', None)
            )
            _, cur_val = training_pipeline(
                model,
                train_x=None,
                val_x=val_dl,
                nepochs=1,
                resume_ckpt=False,
                monitor=config['monitor']
            )
            del val_dl
            gc.collect()
            BEST_LOSS = cur_val if save_best(
                    cur=cur_val,
                    prev=BEST_LOSS,
                    config=config,
                    model=model) else BEST_LOSS
        #os.rmdir("debug_info")

    print('-----------------  done training ----------------------------')
    print("generate evaluation results. ")
    model = cf_model(config)
    model = cf_load(model, config['saved_model_path'])
    model.eval()
    print("Saved model %s Loaded." % config['saved_model_path'])

    print("test search for several keywords.")
    kws = ['analytic', 'innovation', 'technology']
    df_kw_pred = prediction_pipeline(kws,
        config['MTurk_eval_csv'],
        config['eval_data_path'],
        model,
        config['model_name'],
        config['ref_col'],
        config['context_col'],
        config['query_col'],
        batch_size=config['eval_batch_size'],
        limit=config.get('limit', None))
    df_kw_pred.to_csv("cf_predictions.csv", index=False)
    return


if __name__=="__main__":
    main()
