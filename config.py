#!/user/bin/env python
# -*- coding:utf-8 -*-

import os

# train
model_save_path = os.path.abspath('') + "/model/"
dict_path = os.path.abspath('') + "/train_data/dict.txt"
train_data_path = os.path.abspath('') + "/train_data/sentiment_analysis_trainingset.csv"
stop_word_path = os.path.abspath('') + "/train_data/stop.txt"

# validate
validate_data_path = os.path.abspath('') + "/validation/validationset.csv"
predict_result_path = os.path.abspath('') + "/predict_result/"

# test
test_data_path = os.path.abspath('') + "/testdata/testa.csv"
test_data_predict_out_path = os.path.abspath('') + "/predict/testa.csv"

# other
log_path = os.path.abspath('') + "/logs/train.log"
train_data_size = 60000
train_model_num = 1
n_jobs = 10
