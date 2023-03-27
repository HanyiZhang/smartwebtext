#!/bin/bash
a="code"
b="data"
c="fasttext"
d="ais"
e="setup"
f="model"
g="log"
if [ $1 == $g ]
then
  echo "download lightning logs ..."
  mkdir -p lightning_logs_sync
  scp -r shuangludai@34.168.217.130:/home/shuangludai/getwebtext/lightning_logs/* lightning_logs_sync
fi
if [ $1 == $f ]
then
  echo "upload pretrained model ..."
  scp -r data_model/model/*.pth shuangludai@34.168.217.130:/home/shuangludai/getwebtext/data_model/model
fi
if [ $1 == $e ]
then
  echo "upload gcp setup scripts ..."
  scp -r gcp_setup*.sh shuangludai@34.168.217.130:/home/shuangludai
fi
if [ $1 == $a ]
then
  echo "update code to google cloud ..."
  scp -r *.py shuangludai@34.168.217.130:/home/shuangludai/getwebtext
  scp requirements.txt shuangludai@34.168.217.130:/home/shuangludai/getwebtext
  scp -r config shuangludai@34.168.217.130:/home/shuangludai/getwebtext
  scp -r svo shuangludai@34.168.217.130:/home/shuangludai/getwebtext
  scp -r MTurk* shuangludai@34.168.217.130:/home/shuangludai/getwebtext
  scp -r evaluation shuangludai@34.168.217.130:/home/shuangludai/getwebtext
  scp -r professional shuangludai@34.168.217.130:/home/shuangludai/getwebtext
  scp -r env_setup.sh shuangludai@34.168.217.130:/home/shuangludai/getwebtext
fi
if [ $1 == $d ]
then
  echo "update accounting information data ..."
  scp -r ais shuangludai@34.168.217.130:/home/shuangludai/getwebtext
fi
if [ $1 == $b ]
then
  echo "update training validation data ..."
  #scp -r data_model/*cf*.csv shuangludai@34.168.43.23:/home/shuangludai/getwebtext
  #scp -r data_model/data_model_tte_sent.zip shuangludai@34.83.72.82:/home/shuangludai/getwebtext/data_model
  scp -r data_model/data_model_acct_tte_sent.zip shuangludai@34.168.217.130:/home/shuangludai/getwebtext/data_model
  #scp -r data_model/data_model_tte.zip shuangludai@34.83.72.82:/home/shuangludai/getwebtext/data_model
  #scp -r data_model/tte_wgt.zip shuangludai@34.83.72.82:/home/shuangludai/getwebtext/data_model
fi
if [ $1 == $c ]
then
    echo "update fasttext model ..."
    scp -r fasttext/*en*100*.zip shuangludai@34.168.217.130:/home/shuangludai/getwebtext/fasttext
    scp -r fasttext/lid.176.bin shuangludai@34.168.217.130:/home/shuangludai/getwebtext/fasttext
fi


