#!/user/bin/env python
# -*- coding:utf-8 -*-

from data_process import load_data_from_csv, seg_words
from model import TextClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import config
import logging
import numpy as np
from sklearn.externals import joblib
import os
import argparse
from sklearn.metrics import f1_score
import io

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s',
                    filename=config.log_path, filemode='a')
logger = logging.getLogger(__name__)
fh = logging.FileHandler(config.log_path)

# 定义handler的输出格式formatter
formatter = logging.Formatter('%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)

stopwords = [line.strip() for line in io.open(config.stop_word_path, 'r', encoding='utf-8').readlines()]


def train_traffic_mentioned(train_data, validate_data, train_content_segs, validate_data_segs, vectorizer):
    traffic_classifier = dict()
    logger.info("start train traffic mentioned")
    ori_labels = train_data.iloc[0:config.train_data_size, [2, 3, 4]]
    # convert labels ,
    # all the three labels equal -2 means traffic mentioned,covert it to 1
    # else convert it to 0
    train_label = ori_labels.T.sum().abs() // 6
    mentioned_clf = TextClassifier(vectorizer=vectorizer)
    mentioned_clf.fit(train_content_segs, train_label)
    logger.info("begin to validate traffic mentioned model")
    ori_labels = validate_data.iloc[0:, [2, 3, 4]]
    validate_labels = ori_labels.T.sum().abs() // 6
    score = mentioned_clf.get_f1_score(validate_data_segs, validate_labels)
    traffic_classifier["traffic_mentioned"] = mentioned_clf
    logger.info("traffic mentioned model score:%s", str(score))
    if score > 0.8:
        logger.info("save traffic mentioned model")
        model_save_path = config.model_save_path
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        joblib.dump(traffic_classifier, model_save_path + "traffic_mentioned.pkl")
    # logger.info("complete save model")
    # Get columns(model_name) from train data
    columns = train_data.columns.values.tolist()
    scors = dict()
    for index, column in enumerate(columns[2:5]):
        ori_data = train_data_df.iloc[0:config.train_data_size, [1, 2 + index]]
        # filter data with label 1 and -1
        filter_data = ori_data.loc[(ori_data[column] == 1) | (ori_data[column] == -1) | (ori_data[column] == -2)]
        logger.info("begin to seg word for model:%s", column)
        model_content_seg = seg_words(filter_data.iloc[0:, 0])
        # new vectorizer for specific model
        vectorizer_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=10, norm='l2', max_df=0.4,
                                           stop_words=stopwords)
        vectorizer_tfidf.fit(model_content_seg)
        logger.info("complete train feature extraction model:%s", column)
        logger.info("vocab shape: %s" % np.shape(vectorizer_tfidf.vocabulary_.keys()))
        positive_clf = TextClassifier(vectorizer=vectorizer_tfidf)

        positive_label = filter_data.iloc[0:, 1]
        logger.info("start train %s model" % column)
        positive_clf.fit(model_content_seg, positive_label)
        logger.info("complete train %s model" % column)
        traffic_classifier[column] = positive_clf

        # validate model
        # predict by mentioned_clf
        logger.info("validate model:%s", column)
        predict = mentioned_clf.predict(validate_data_segs)
        predict = predict * -2
        # predict rest validate data with positive_clf
        for v_index, v_content_seg in enumerate(validate_data_segs):
            if predict[v_index] == 0:
                predict[v_index] = positive_clf.predict([v_content_seg])
        final_score = f1_score(validate_data.iloc[0:, [2 + index]], predict, average='macro')
        scors[column] = final_score
        logger.info("score for model:%s is %s ", column, str(final_score))
    str_score = "\n"
    score = np.mean(list(scors.values()))
    for column in columns[2:5]:
        str_score = str_score + column + ":" + str(scors[column]) + "\n"

    logger.info("f1_scores: %s\n" % str_score)
    logger.info("f1_score: %s" % score)
    logger.info("complete validate model")


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
    train_data_size = config.train_data_size
    content_train = train_data_df.iloc[0:train_data_size, 1]

    logger.info("start seg train data")
    content_train = seg_words(content_train)
    logger.info("complete seg train data")

    # load validate model
    content_validate = validate_data_df.iloc[:, 1]

    logger.info("start seg validate data")
    content_validate = seg_words(content_validate)
    logger.info("complete seg validate data")
    # train traffic mentioned

    columns = train_data_df.columns.values.tolist()

    logger.info("start train feature extraction")
    # vectorizer_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=5, norm='l2', stop_words=stopwords)
    vectorizer_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=5, norm='l2', max_df=0.7,
                                       stop_words=stopwords)
    vectorizer_tfidf.fit(content_train)
    logger.info("complete train feature extraction models")
    logger.info("vocab shape: %s" % np.shape(vectorizer_tfidf.vocabulary_.keys()))

    train_traffic_mentioned(train_data_df, validate_data_df, content_train, content_validate, vectorizer_tfidf)

    # # model train
    # logger.info("start train model")
    # classifier_dict = dict()
    # i = 0
    # for column in columns[2:]:
    #     label_train = train_data_df.iloc[0:train_data_size, 2 + i]
    #     text_classifier = TextClassifier(vectorizer=vectorizer_tfidf)
    #     logger.info("start train %s model" % column)
    #     text_classifier.fit(content_train, label_train)
    #     logger.info("complete train %s model" % column)
    #     classifier_dict[column] = text_classifier
    #     i += 1
    #
    # logger.info("complete train model")
    #
    # logger.info("start validate model")
    # f1_score_dict = dict()
    # for column in columns[2:]:
    #     label_validate = validate_data_df[column]
    #     text_classifier = classifier_dict[column]
    #     score = text_classifier.get_f1_score(content_validate, label_validate)
    #     f1_score_dict[column] = score
    #
    # score = np.mean(list(f1_score_dict.values()))
    # str_score = "\n"
    # for column in columns[2:]:
    #     str_score = str_score + column + ":" + str(f1_score_dict[column]) + "\n"
    #
    # logger.info("f1_scores: %s\n" % str_score)
    # logger.info("f1_score: %s" % score)
    # logger.info("complete validate model")
    #
    # # save model
    # logger.info("start save model")
    # model_save_path = config.model_save_path
    # if not os.path.exists(model_save_path):
    #     os.makedirs(model_save_path)
    #
    # joblib.dump(classifier_dict, model_save_path + model_name)
    # logger.info("complete save model")
