import pandas as pd
from embedding import LocalTopicAsEmbedding as Embedding
from preprocessor import pipeline as preproc_pipeline
from preprocessor import run_pipeline as preprocess

RAW_DATA_PATH = "data_model/ner_cleaned_title.csv"
CACHED_DATA_PATH = 'data_model/cached_acct_training_data.csv'
MODEL_PATH = 'config/topic_emb.yaml'

def main():
    load_cached_preproc_data= False
    df = pd.read_csv(RAW_DATA_PATH)
    preprocessing_fns =  preproc_pipeline(df, load_cached_preproc_data)
    df = preprocess(df, preprocessing_fns)
    df.to_csv(CACHED_DATA_PATH)
    trm_emb=Embedding(MODEL_PATH)
    trm_emb.train(df[['Company','Text']])
    return


if __name__=="__main__":
    main()
