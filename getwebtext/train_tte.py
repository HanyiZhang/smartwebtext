import json
import glob
import os
import gc
import pandas as pd
from preprocessor import tte_pipeline as preproc_pipeline
from preprocessor import run_pipeline as preprocess
from preprocessor import load_cached_preprocessed_csv
from train_utils import training_pipeline, read_config
from train_utils import load as cf_load
from data_utils import get_context_csv_data_loader, randsample_df_by_group_rate
from cf import train_sentence_features as train_features
from cf import get_tte_prediction_fn, get_unflatten_fn


def extract_ref_subset(df, config, ref_subset, split):
    df = df[df[config['ref_col']].isin(ref_subset)]
    df = df.drop_duplicates(subset=[config["query_col"], config["ref_col"]], keep='last')
    print("%s %d companies." % (split,len(set(df[config['ref_col']])) ) )
    print(df[config['ref_col']].value_counts())
    return df


def train_test_split(df, config, keep_cols, rate=0.002):
    df['positives'] = df[config["score_col"]] > 0
    df_val = randsample_df_by_group_rate(df, ['Company', "positives"], rate=rate)
    df_val = df_val[~df_val.index.duplicated(keep='first')]
    df_val = df_val[keep_cols]
    df_val = df_val[~df_val.isnull().any(axis=1)]
    df = df[~df.index.isin(df_val.index)]
    df = df[keep_cols]
    # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df[~df.isnull().any(axis=1)]
    return df, df_val

def get_eval_company(eval_fpath):
    eval_company = pd.read_csv(eval_fpath)
    rename_dict = {col: col.lower() for col in eval_company.columns}
    eval_company = eval_company.rename(rename_dict, axis=1)
    eval_company = eval_company['company']
    return eval_company


def main():
    config = read_config("config/acct_tte_sent_small.yaml")
    if config['model_name'] in ['tte_small', 'tte_sent_small', 'acct_tte_sent_small']:
        from cf import TTEModel as cf_model
        from cf import train_phrase_features as train_features
    elif config['model_name'] in ['tte_bert_item_small']:
        from cf import TTEBertItem as cf_model
        from cf import train_phrase_features as train_features
    elif config['model_name'] in ['tte_sent_small_semisup']:
        from cf import SemiSupTTEModel as cf_model
    else:
        print('model not found')
        return
    #config = read_config(sys.argv[1])
    keep_cols = [
        config["score_col"],
        config["query_col"],
        config["ref_col"]
    ]
    if not config.get("skip_prep_data", False):
        print('prepare dataset for tte training. ')
        df = pd.read_csv(config['raw_data_path'])
        preprocessing_fns = preproc_pipeline(df, load_path="", vocab_path=config['vocab_path'])
        df = preprocess(df, preprocessing_fns)
        df.to_csv(config['preproc_data_path'], index=False)
        df = load_cached_preprocessed_csv(data_path=config['preproc_data_path'])
        df_train, df_val = train_test_split(df, config, keep_cols, rate=config['train_test_split_ratio'])
        eval_company = get_eval_company(config['MTurk_eval_csv'])
        df_train = extract_ref_subset(df_train, config, eval_company, 'training')
        df_train.to_csv(config["train_data_path"], index=False)
        df_val = extract_ref_subset(df_val, config, eval_company, 'validation')
        df_val.to_csv(config["val_data_path"], index=False)
        item_vocab = {v: i for i, v in enumerate(sorted(set(df_train[config['ref_col']])))}
        with open(config['item_vocab_path'], 'w') as f:
            json.dump(item_vocab, f)
        del item_vocab
        del df
        del df_train
        del df_val
        gc.collect()

    model = cf_model(config)
    print("model initialized. " )
    if not config.get('skip_training', False):
        print("create training -validation dataloader")
        train_dl = get_context_csv_data_loader(
            config['train_data_path'],
            train_features,
            batch_size=config['batch_size'],
            clear_cache=config['clear_cache'],
            shuffle=True,
            sep=',',
            max_line=10 ** 7,
            limit=config.get('limit', None))
        val_dl = get_context_csv_data_loader(
            config['val_data_path'], train_features,
            batch_size=config['batch_size'],
            clear_cache=config['clear_cache'],
            shuffle=False,
            sep=',',
            max_line=10 ** 7,
            limit=config.get('limit', None)
        )
        logger_dir = config.get("logger_dir", "./lightning_logs")
        os.makedirs(logger_dir, exist_ok=True)
        print(" start training ...")
        model, _ = training_pipeline(
            model,
            train_dl,
            val_x=val_dl,
            nepochs=config['epochs'],
            resume_ckpt=config['resume_ckpt'],
            model_name=config['model_name'],
            monitor=config['monitor'],
            logger_path=config['logger_dir']
        )
    list_of_files = glob.glob(os.path.join(config['logger_dir'], '%s-epoch*.ckpt' % config['model_name']))  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print("found checkpoint %s" % latest_file)
    print('-----------------  done training ----------------------------')
    os.makedirs('evaluation', exist_ok=True)
    print("generate evaluation results. ")
    model = cf_model(config)
    model = cf_load(model, latest_file)
    model.eval()
    print("checkpoint %s Loaded." % latest_file)
    kws = ['analytics', 'innovation', 'technology']
    print("test search for keywords: ", kws)
    unflatten_fn = get_unflatten_fn(
        config['ref_col'], config['query_col'], config['score_col'], config['model_name'])
    #eval_company = pd.read_csv(config['MTurk_eval_csv'])['company'].tolist()
    eval_company = get_eval_company(config['MTurk_eval_csv'])
    eval_company = eval_company.tolist()
    predict_fn = get_tte_prediction_fn(
        refs=eval_company,
        model= model,
        ref_col=config['ref_col'],
        query_col=config['query_col'],
        score_col=config['score_col'],
        unflatten_fn=unflatten_fn
    )
    df_kw_pred = predict_fn(kws=kws, unflatten=True)
    df_kw_pred.to_csv("evaluation/%s_eval_predictions.csv" % config['model_name'], index=False)
    print("---------------------generate predictions on validation set -------------------------------")
    val_data = pd.read_csv(config['val_data_path'])
    #val_kws = sorted(set(val_kws))
    df_val_preds = []
    for i, eval_ref in enumerate(eval_company):
        print('%d/%d %s' % (i, len(eval_company), eval_ref))
        kws_id = val_data[config['ref_col']] == eval_ref
        if kws_id.sum() == 0:
            continue
        val_kws = val_data[config['query_col']][kws_id].tolist()
        predict_fn = get_tte_prediction_fn(
            refs=[eval_ref],
            model=model,
            ref_col=config['ref_col'],
            query_col=config['query_col'],
            score_col=config['score_col'],
            unflatten_fn=unflatten_fn
            )
        df_val_pred = predict_fn(kws=val_kws, unflatten=False)
        df_val_preds.append(df_val_pred)
    df_val_preds = pd.concat(df_val_preds)
    df_val_preds.to_csv('evaluation/%s_val_predictions.csv' % config['model_name'], index=False)
    return


if __name__=="__main__":
    main()
    """
    train_data_path = config['tte_train_data_path']
    if not config.get("split_data", False):
        print('remove validation data from training set.')
        df_train = pd.read_csv(config['tte_train_data_path'])
        df_train = df_train[keep_cols]
        df_train.set_index([config["query_col"], config["ref_col"]], inplace=True)
        df_val = pd.read_csv(config['tte_val_data_path'])
        df_val = df_val[keep_cols]
        df_val.set_index([config["query_col"], config["ref_col"]], inplace=True)
        val_index = df_train.index.intersection(df_val.index)
        print("%d duplicated samples from training found in the validation set." % len(val_index))
        df_train = df_train.drop(val_index)
        train_data_path = config['tte_train_data_path'].replace('.csv', '.split.csv')
        df_train.to_csv(train_data_path)
        """
