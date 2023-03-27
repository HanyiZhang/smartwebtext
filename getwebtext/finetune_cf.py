import gc
import pandas as pd
import os
from train_utils import training_pipeline, read_config
from train_utils import load as cf_load
from train_utils import save as cf_save
from data_utils import  randsample_df_by_group_size, get_context_csv_data_loader
from cf import train_phrase_features, prediction_pipeline
from cf import TTEContextModel as cf_model


def finetuned_path(x: str, prefix: str):
    a, b = list(os.path.split(x))
    b = prefix+b
    return os.path.join(a,b)


def main():
    config = read_config("config/finetune_cf.yaml")
    keep_cols = [
        config["score_col"],
        config["query_col"],
        config["context_col"],
        config["ref_col"]
    ]
    df_val = pd.read_csv(config['val_data_path'])
    df_val['positives'] = df_val[config["score_col"]] > 0
    df_val = randsample_df_by_group_size(df_val, ['Company', "positives"], size=10)
    df_val = df_val[~df_val.index.duplicated(keep='first')]
    df_val = df_val[keep_cols]
    val_data_path = finetuned_path(config["val_data_path"],config['path_prefix'])
    df_val.to_csv(val_data_path)
    model = cf_model(config)
    print("model initialized. " )
    if not config.get('skip_training', False):
        print("create training -validation dataloader")
        val_dl = get_context_csv_data_loader(
            val_data_path, train_phrase_features,
            batch_size=config['batch_size'],
            clear_cache=config['clear_cache'],
            shuffle=False,
            sep=',',
            max_line=10 ** 7,
            limit=config.get('limit', None)
        )
        train_dl = get_context_csv_data_loader(
            config['train_data_path'],
            train_phrase_features,
            batch_size=config['batch_size'],
            clear_cache=config['clear_cache'],
            shuffle=True,
            sep=',',
            max_line=10 ** 7,
            limit=config.get('limit', None))

        logger_dir = config.get("logger_dir", "./lightning_logs")
        os.makedirs(logger_dir, exist_ok=True)
        print(" start training ...")
        model, _ = training_pipeline(
            model,
            train_dl,
            val_dl,
            nepochs=config['epochs'],
            enable_ckpt=True,
            resume_ckpt=config.get("resume_ckpt", False),
            monitor=config['monitor'])
        model_save_path = finetuned_path(config['saved_model_path'], config['path_prefix'])
        cf_save(model, model_save_path)
        print("Finetuned model %s Saved." % config['saved_model_path'])
        del train_dl
        del val_dl
        gc.collect()

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
                                     config['query_col'])
    df_kw_pred.to_csv("cf_predictions.csv", index=False)

    return


if __name__=="__main__":
    main()

