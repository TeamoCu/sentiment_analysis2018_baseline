#!/user/bin/env python
# -*- coding:utf-8 -*-
from sklearn.metrics import f1_score

from data_process import load_data_from_csv, seg_words
from model import TextClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import config
import numpy as np
from sklearn.externals import joblib
import os
import argparse
import io
from Logger import logger

stopwords = [line.strip() for line in io.open(config.stop_word_path, 'r', encoding='utf-8').readlines()]


def train_traffic_mentioned(train_data, validate_data):
    traffic_classifier = dict()
    logger.info("start train traffic mentioned")
    train_data_size = 10000
    ori_labels = train_data.iloc[0:train_data_size, [2, 3, 4]]
    # convert labels ,
    # all the three labels equal -2 means traffic mentioned,covert it to 1
    # else convert it to 0
    train_label = ori_labels.T.sum().abs() // 6

    content_train = train_data_df.iloc[0:train_data_size, 1]
    # seg and vectorizer train data
    logger.debug("start seg train data")
    train_content_segs = seg_words(content_train)
    logger.debug("complete seg train data")
    vectorizer_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=5, norm='l2', max_df=0.7,
                                       stop_words=stopwords)
    vectorizer_tfidf.fit(train_content_segs)
    logger.debug("vocab shape: %s" % np.shape(vectorizer_tfidf.vocabulary_.keys()))
    logger.debug("begin to train data")
    mentioned_clf = TextClassifier(vectorizer=vectorizer_tfidf)
    mentioned_clf.fit(train_content_segs, train_label)
    logger.debug("begin to validate traffic mentioned model")
    # load validate model
    content_validate = validate_data_df.iloc[:, 1]
    logger.debug("start seg validate data")
    validate_data_segs = seg_words(content_validate)
    logger.debug("complete seg validate data")
    ori_labels = validate_data.iloc[0:, [2, 3, 4]]
    validate_labels = ori_labels.T.sum().abs() // 6
    score = mentioned_clf.get_f1_score(validate_data_segs, validate_labels)
    logger.info("validate done! traffic mentioned model score:%s", str(score))

    traffic_classifier["traffic_mentioned"] = mentioned_clf
    if score > 0.8:
        logger.info("save traffic mentioned model")
        model_save_path = config.model_save_path
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        joblib.dump(traffic_classifier, model_save_path + "traffic_mentioned.pkl")
    return mentioned_clf


def train_traffic_model(train_data, validate_data):
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
    vectorizer_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=10, norm='l2', max_df=0.4,
                                       stop_words=stopwords)
    vectorizer_tfidf.fit(model_content_seg)
    logger.debug("complete train feature extraction")
    logger.info("vocab shape: %s" % np.shape(vectorizer_tfidf.vocabulary_.keys()))
    # filter validate data
    validate_data = validate_data.loc[
        (validate_data[columns[2]] != -2) | (validate_data[columns[3]] != -2) | (validate_data[columns[4]] != -2)]
    content_validate = validate_data.iloc[0:, 1]
    validate_data_segs = seg_words(content_validate)

    scors = dict()
    for index, column in enumerate(columns[2:5]):
        positive_clf = TextClassifier(vectorizer=vectorizer_tfidf)

        positive_label = filter_data.iloc[0:, 1 + index]
        logger.info("start train %s model" % column)
        positive_clf.fit(model_content_seg, positive_label)
        logger.info("complete train %s model" % column)
        final_score = positive_clf.get_f1_score(validate_data_segs, validate_data.iloc[0:, [2 + index]])
        scors[column] = final_score
        logger.info("score for model:%s is %s ", column, str(final_score))
        logger.info("save traffic mentioned model")
        joblib.dump(positive_clf, config.model_save_path + column + ".pkl")
    str_score = "\n"
    score = np.mean(list(scors.values()))
    for column in columns[2:5]:
        str_score = str_score + column + ":" + str(scors[column]) + "\n"

    logger.info("f1_scores: %s\n" % str_score)
    logger.info("f1_score: %s" % score)
    logger.info("complete validate model")


def validate_traffic(validate_data, columns, mentioned_clf, clfs):
    logger.info("Begin validate traffic")
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
            if tmp_predict[v_index] == 0:
                tmp_predict[v_index] = clfs[column].predict([v_content_seg])
            file.write(str(tmp_predict[v_index]) + u"\n")
        file.close()
        score = f1_score(validate_data[column], tmp_predict, average='macro')
        scores[column] = score

    str_score = "\n"
    score = np.mean(list(scores.values()))
    for column in columns:
        str_score = str_score + column + ":" + str(scores[column]) + "\n"

    logger.info("f1_scores: %s\n" % str_score)
    logger.info("f1_score: %s" % score)
    logger.info("complete validate model")
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', '--model_name', type=str, nargs='?',
                        help='the name of model')

    args = parser.parse_args()
    model_name = args.model_name
    if not model_name:
        model_name = "model_dict.pkl"

    # load train data
    logger.info("########################################")
    logger.info("start load data")
    logger.info("########################################")
    train_data_df = load_data_from_csv(config.train_data_path)
    validate_data_df = load_data_from_csv(config.validate_data_path)
    train_traffic_mentioned(train_data_df, validate_data_df)
    train_traffic_model(train_data_df, validate_data_df)
    columns = train_data_df.columns.values.tolist()[2:5]
    m_clf = joblib.load(config.model_save_path + "traffic_mentioned.pkl")

    clf_dict = dict()
    for column in columns:
        clf = joblib.load(config.model_save_path + column + ".pkl")
        clf_dict[column] = clf

    validate_traffic(validate_data_df, columns, m_clf["traffic_mentioned"], clf_dict)
