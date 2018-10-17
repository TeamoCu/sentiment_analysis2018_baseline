#!/user/bin/env python
# -*- coding:utf-8 -*-
import io
import os

import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

import config
from Logger import logger
from data_process import load_data_from_csv, seg_words
from model import TextClassifier

stopwords = [line.strip() for line in io.open(config.stop_word_path, 'r', encoding='utf-8').readlines()]
vec_name = "tf-idf.vec"
models = [('traffic', 2, 4), ('service', 5, 8), ('price', 9, 11), ('enviorment', 12, 15)]


def train_mentioned_model(train_data, train_segs, validate_data, validate_segs, vectorizer, train_model):
    model_name = train_model[0]
    start = train_model[1]
    end = train_model[2]
    logger.info("start train %s mentioned", model_name)
    train_data_size = config.train_data_size
    sum_label_val = (end - start + 1) * 2
    column_list = range(start, end + 1)
    ori_labels = train_data.iloc[0:train_data_size, column_list]
    # convert labels ,
    # all the three labels equal -2 means mentioned this item,covert it to 1
    # else convert it to 0
    train_label = ori_labels.T.sum().abs() // sum_label_val
    logger.debug("begin to train data")
    cw = [{0: w, 1: x} for w in range(1, 10), for x in range(1, 5)]
    mentioned_clf = TextClassifier(vectorizer=vectorizer, class_weight=cw)
    mentioned_clf.fit(train_segs, train_label)
    logger.debug("begin to validate %s mentioned model", model_name)
    # load validate model
    ori_labels = validate_data.iloc[0:, column_list]
    validate_labels = ori_labels.T.sum().abs() // sum_label_val
    y_pre = mentioned_clf.predict(validate_segs)
    report(validate_labels, y_pre)
    score = f1_score(validate_labels, y_pre, average="macro")
    logger.info("validate done! %s mentioned model score:%s", model_name, str(score))

    if score > 0.8:
        logger.info("save %s mentioned model", model_name)
        model_save_path = config.model_save_path
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        joblib.dump(mentioned_clf, model_save_path + model_name + "_mentioned.pkl", compress=3)
    return mentioned_clf


def train_specific_model(train_data):
    columns = train_data.columns.values.tolist()
    content_segments = seg_words(train_data.content.iloc[0:config.train_data_size])
    logger.debug("seg train content")
    vectorizer = joblib.load(config.model_save_path + vec_name)
    logger.debug("load vectorizer")
    validate_data_df = load_data_from_csv(config.validate_data_path)
    validata_segs = seg_words(validate_data_df.content)
    logger.debug("seg validate content")
    scores = dict()
    for model_name in columns[:-1]:
        logger.info("begin to train %s model", model_name)
        cw = [{-2: a, -1: b, 0: w, 1: x} for a in range(1, 3), for b in range(1, 8) for w in range(1, 8), for x in
              range(1, 8)]
        positive_clf = TextClassifier(vectorizer=vectorizer, class_weight=cw)
        y_label = train_data[model_name]
        positive_clf.fit(content_segments, y_label)

        y_pre = positive_clf.predict(validata_segs)
        y_true = validata_segs[model_name]
        report(y_true, y_pre)
        score = f1_score(y_true, y_pre, average="macro")
        logger.info("score for model:%s is %s ", model_name, str(score))
        scores[model_name] = score
        joblib.dump(positive_clf, config.model_save_path + model_name + ".pkl", compress=3)
    score = np.mean(list(scores.values()))
    logger.info("f1_scores: %s" % score)


def report(y_true, tmp_predict, model_name=""):
    report_str = classification_report(y_true, tmp_predict)
    logger.info("Report for model %s:\n%s", model_name, report_str)


#
# def validate_model(validate_data, columns, mentioned_clf, clfs):
#     logger.info("Begin validate")
#     content_validate = validate_data.iloc[:, 1]
#     validate_data_segs = seg_words(content_validate)
#     logger.debug("seg validate data done")
#
#     scores = dict()
#     predict = mentioned_clf.predict(validate_data_segs)
#     predict = predict * -2
#     for column in columns:
#         logger.debug("predict:%s", column)
#         tmp_predict = predict.copy()
#         file = io.open(config.predict_result_path + column + ".txt", "w", encoding="utf-8")
#         for v_index, v_content_seg in enumerate(validate_data_segs):
#             proba_str = "\t"
#             if tmp_predict[v_index] == 0:
#                 tmp_predict[v_index] = clfs[column].predict([v_content_seg])
#                 proba = clfs[column].predict_proba([v_content_seg])
#                 proba_str = str(proba)
#             file.write(str(tmp_predict[v_index]) + proba_str + u"\n")
#         file.close()
#         report(validate_data[column], tmp_predict)
#         score = f1_score(validate_data[column], tmp_predict, average='macro')
#         scores[column] = score
#
#     str_score = "\n"
#     score = np.mean(list(scores.values()))
#     for column in columns:
#         str_score = str_score + column + ":" + str(scores[column]) + "\n"
#
#     logger.info("f1_scores: %s\n" % str_score)
#     logger.info("f1_score: %s" % score)
#     logger.info("complete validate model")


def vectorizer():
    logger.info("start to vectorizer content")
    train_data = load_data_from_csv(config.train_data_path)
    content_segs = seg_words(train_data.iloc[0:config.train_data_size, 1])
    tf_idf = TfidfVectorizer(ngram_range=(1, 6), min_df=2, norm="l2", max_df=0.3)
    tf_idf.fit(content_segs)
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    joblib.dump(tf_idf, config.model_save_path + vec_name, compress=True)
    logger.info("succes to save vectorizer")


def train_mentioned():
    logger.info("########################################")
    logger.info("start train mentioned models")
    logger.info("########################################")
    # load train data
    train_data_df = load_data_from_csv(config.train_data_path)
    validate_data_df = load_data_from_csv(config.validate_data_path)

    content_train = train_data_df.iloc[0:config.train_data_size, 1]
    logger.debug("start seg train data")
    train_content_segs = seg_words(content_train)

    logger.debug("start seg validate data")
    content_validate = validate_data_df.iloc[0:, 1]
    validate_segs = seg_words(content_validate)
    logger.debug("load vectorizer")
    vectorizer_tfidf = joblib.load(config.model_save_path + vec_name)

    for model in models:
        train_mentioned_model(train_data_df, train_content_segs, validate_data_df, validate_segs, vectorizer_tfidf,
                              model)


def filter_data(data, model):
    columns = data.columns.values.tolist()
    lable_sum = (model[2] - model[1] + 1) * -2
    target_columns = columns[model[1]:model[2] + 1].append('content')
    logger.debug("filter data for columns:%s", target_columns)
    target_data = data.loc[target_columns]
    target_data['sum'] = target_data[columns[model[1]:model[2] + 1]].T.sum().T
    target_data = target_data.iloc[target_data['sum'] > lable_sum, target_columns]
    return target_data


def train_model():
    logger.info("########################################")
    logger.info("start train models")
    logger.info("########################################")
    train_data_df = load_data_from_csv(config.train_data_path)
    for model in models:
        data_to_train = filter_data(train_data_df, model)
        train_specific_model(data_to_train)


if __name__ == '__main__':
    # train_mentioned()
    train_model()
    # validate traffic
    # train_traffic(train_data_df, validate_data_df)
    # validate_traffic()
    # train service
    # train_service(train_data_df, validate_data_df, 'service')
    # validate('service', 5, 8)
