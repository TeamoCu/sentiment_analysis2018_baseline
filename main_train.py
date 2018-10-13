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


def train_mentioned(train_data, validate_data, model_name, start, end):
    logger.info("start train %s mentioned", model_name)
    train_data_size = config.train_data_size
    sum_label_val = (end - start + 1) * 2
    column_list = range(start, end + 1)
    ori_labels = train_data.iloc[0:train_data_size, column_list]
    # convert labels ,
    # all the three labels equal -2 means mentioned this item,covert it to 1
    # else convert it to 0
    train_label = ori_labels.T.sum().abs() // sum_label_val

    content_train = train_data_df.iloc[0:train_data_size, 1]
    # seg and vectorizer train data
    logger.debug("start seg train data")
    train_content_segs = seg_words(content_train)
    vectorizer_tfidf = joblib.load(config.model_save_path + vec_name)
    logger.debug("vocab shape: %s" % np.shape(vectorizer_tfidf.vocabulary_.keys()))
    logger.debug("begin to train data")
    mentioned_clf = TextClassifier(vectorizer=vectorizer_tfidf)
    mentioned_clf.fit(train_content_segs, train_label)
    logger.debug("begin to validate %s mentioned model", model_name)
    # load validate model
    content_validate = validate_data_df.iloc[:, 1]
    logger.debug("start seg validate data")
    validate_data_segs = seg_words(content_validate)
    logger.debug("complete seg validate data")
    ori_labels = validate_data.iloc[0:, column_list]
    validate_labels = ori_labels.T.sum().abs() // sum_label_val
    y_pre = mentioned_clf.predict(validate_data_segs)
    report(validate_labels, y_pre)
    score = f1_score(validate_labels, y_pre, average="macro")
    logger.info("validate done! %s mentioned model score:%s", model_name, str(score))

    if score > 0.8:
        logger.info("save %s mentioned model", model_name)
        model_save_path = config.model_save_path
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        joblib.dump(mentioned_clf, model_save_path + model_name + "_mentioned.pkl")
    return mentioned_clf


def train_traffic(train_data, validate_data):
    logger.info("begin to train traffic model")
    # Get columns(model_name) from train data
    columns = train_data.columns.values.tolist()
    ori_df = train_data_df.iloc[0:config.train_data_size, [1, 2, 3, 4]]
    # filter data not mentioned traffic
    filter_data = ori_df.loc[(ori_df[columns[2]] != -2) | (ori_df[columns[3]] != -2) | (ori_df[columns[4]] != -2)]
    logger.info("filter data mentioned traffic,data size:%d", filter_data.size / 4)
    logger.debug("begin to seg word for traffic model")
    model_content_seg = seg_words(filter_data.iloc[0:, 0])
    logger.debug("end to seg word")
    # new vectorizer for specific model
    vectorizer_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 6), min_df=3, norm='l2', max_df=0.4,
                                       stop_words=stopwords)
    vectorizer_tfidf.fit(model_content_seg)
    logger.debug("complete train feature extraction")
    logger.info("vocab shape: %s" % np.shape(vectorizer_tfidf.vocabulary_.keys()))
    # filter validate data
    validate_data = validate_data.loc[
        (validate_data[columns[2]] != -2) | (validate_data[columns[3]] != -2) | (validate_data[columns[4]] != -2)]
    content_validate = validate_data.iloc[0:, 1]
    validate_data_segs = seg_words(content_validate)

    scores = dict()
    for index, column in enumerate(columns[2:5]):
        logger.info("start train %s model" % column)
        positive_label = filter_data.iloc[0:, 1 + index]
        positive_clf = TextClassifier(vectorizer=vectorizer_tfidf, class_weight={-1: 3, 0: 8})
        positive_clf.fit(model_content_seg, positive_label)
        logger.info("complete train %s model" % column)
        final_score = positive_clf.get_f1_score(validate_data_segs, validate_data.iloc[0:, [2 + index]])
        scores[column] = final_score
        logger.info("score for model:%s is %s ", column, str(final_score))
        joblib.dump(positive_clf, config.model_save_path + column + ".pkl")
        logger.debug("save model %s", column)
    str_score = "\n"
    score = np.mean(list(scores.values()))
    for column in columns[2:5]:
        str_score = str_score + column + ":" + str(scores[column]) + "\n"

    logger.info("f1_scores: %s\n" % str_score)
    logger.info("f1_score: %s" % score)
    logger.info("complete validate model")


def train_service(train_data, validate_data, model_name):
    logger.info("begin to train %s model", model_name)
    # Get columns(model_name) from train data
    columns = train_data.columns.values.tolist()
    ori_df = train_data_df.iloc[0:config.train_data_size, [1, 5, 6, 7, 8]]
    # filter data not mentioned traffic
    filter_data = ori_df.loc[(ori_df[columns[5]] != -2) | (ori_df[columns[6]] != -2) | (ori_df[columns[7]] != -2) | (
            ori_df[columns[8]] != -2)]
    logger.info("filter data mentioned %s,data size:%d", model_name, filter_data.size / 5)
    logger.debug("begin to seg word for %s model", model_name)
    model_content_seg = seg_words(filter_data.iloc[0:, 0])
    logger.debug("end to seg word")
    # new vectorizer for specific model
    vectorizer_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 6), min_df=3, norm='l2', max_df=0.4,
                                       stop_words=stopwords)
    vectorizer_tfidf.fit(model_content_seg)
    logger.debug("complete train feature extraction")
    logger.info("vocab shape: %s" % np.shape(vectorizer_tfidf.vocabulary_.keys()))
    # filter validate data
    validate_data = validate_data.loc[
        (ori_df[columns[5]] != -2) | (ori_df[columns[6]] != -2) | (ori_df[columns[7]] != -2) | (
                ori_df[columns[8]] != -2)]
    content_validate = validate_data.iloc[0:, 1]
    validate_data_segs = seg_words(content_validate)

    scores = dict()
    for index, column in enumerate(columns[5:9]):
        logger.info("start train %s model" % column)
        positive_label = filter_data.iloc[0:, 1 + index]
        # positive_clf = TextClassifier(vectorizer=vectorizer_tfidf, class_weight={-1: 5, 0: 10})
        positive_clf = TextClassifier(vectorizer=vectorizer_tfidf)
        positive_clf.fit(model_content_seg, positive_label)
        logger.info("complete train %s model" % column)
        final_score = positive_clf.get_f1_score(validate_data_segs, validate_data.iloc[0:, [2 + index]])
        scores[column] = final_score
        logger.info("score for model:%s is %s ", column, str(final_score))
        joblib.dump(positive_clf, config.model_save_path + column + ".pkl")
        logger.debug("save model %s", column)
    str_score = "\n"
    score = np.mean(list(scores.values()))
    for column in columns[2:5]:
        str_score = str_score + column + ":" + str(scores[column]) + "\n"

    logger.info("f1_scores: %s\n" % str_score)
    logger.info("f1_score: %s" % score)
    logger.info("complete validate model")


def report(y_true, tmp_predict, model_name=""):
    report_str = classification_report(y_true, tmp_predict)
    logger.info("Report for model %s:\n%s", model_name, report_str)


def validate_model(validate_data, columns, mentioned_clf, clfs):
    logger.info("Begin validate")
    content_validate = validate_data.iloc[:, 1]
    validate_data_segs = seg_words(content_validate)
    logger.debug("seg validate data done")

    scores = dict()
    predict = mentioned_clf.predict(validate_data_segs)
    predict = predict * -2
    for column in columns:
        logger.debug("predict:%s", column)
        tmp_predict = predict.copy()
        file = io.open(config.predict_result_path + column + ".txt", "w", encoding="utf-8")
        for v_index, v_content_seg in enumerate(validate_data_segs):
            proba_str = "\t"
            if tmp_predict[v_index] == 0:
                tmp_predict[v_index] = clfs[column].predict([v_content_seg])
                proba = clfs[column].predict_proba([v_content_seg])
                proba_str = str(proba)
            file.write(str(tmp_predict[v_index]) + proba_str + u"\n")
        file.close()
        report(validate_data[column], tmp_predict)
        score = f1_score(validate_data[column], tmp_predict, average='macro')
        scores[column] = score

    str_score = "\n"
    score = np.mean(list(scores.values()))
    for column in columns:
        str_score = str_score + column + ":" + str(scores[column]) + "\n"

    logger.info("f1_scores: %s\n" % str_score)
    logger.info("f1_score: %s" % score)
    logger.info("complete validate model")


def validate(model_name, columns_start, columns_end):
    columns = validate_data_df.columns.values.tolist()[columns_start:columns_end]
    m_clf = joblib.load(config.model_save_path + +model_name + "_mentioned.pkl")
    clf_dict = dict()
    for column in columns:
        clf = joblib.load(config.model_save_path + column + ".pkl")
        clf_dict[column] = clf
    validate_model(validate_data_df, columns, m_clf, clf_dict)


def vectorizer():
    logger.info("start to vectorizer content")
    train_data = load_data_from_csv()
    content_segs = seg_words(train_data.iloc[0:, 1])
    tf_idf = TfidfVectorizer(ngram_range=(1, 6), min_df=2, norm="l2", max_df=0.3)
    tf_idf.fit(content_segs)
    joblib.dump(tf_idf, config.model_save_path + vec_name, compress=True)
    logger.info("succes to save vectorizer")


if __name__ == '__main__':
    logger.info("Report for model \s \s", "1", "2")
    logger.info("########################################")
    logger.info("start train")
    logger.info("########################################")
    # load train data
    train_data_df = load_data_from_csv(config.train_data_path)
    validate_data_df = load_data_from_csv(config.validate_data_path)

    models = [('service', 5, 8), ('traffic', 2, 4), ('price', 9, 11), ('enviorment', 12, 15)]
    for model in models:
        train_mentioned(train_data_df, validate_data_df, model[0], model[1], model[2])
        # validate(model[0], model[1], model[2])

    # validate traffic
    # train_traffic(train_data_df, validate_data_df)
    # validate_traffic()
    # train service
    # train_service(train_data_df, validate_data_df, 'service')
    # validate('service', 5, 8)
