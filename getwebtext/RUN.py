import os

python_cmd = "python"
# data scrapping - generate 'data_model/extracted_text.txt'
os.system(' '.join([python_cmd, "webscrapping.py"]))
# prepreprocessing - generate "data_model/ner_cleaned_title.csv"
os.system(' '.join([python_cmd, "data_preparation.py"]))
# TRM modeling - generate "data_model/model/company_trm"
os.system(' '.join([python_cmd, "train.py"]))
# Model inference - generate "eval_accoutingInfo_dataset.csv"
os.system(' '.join([python_cmd, "search.py"]))
# Model Evaluation using Amazon Turk batch labels MTurkBatch/Batch_*_batch_results.csv
os.system(' '.join([python_cmd, "renderTurk.py"]))