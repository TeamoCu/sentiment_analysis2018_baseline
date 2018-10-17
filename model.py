#!/user/bin/env python
# -*- coding:utf-8 -*-

from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from Logger import logger
import config


class TextClassifier:

    def __init__(self, vectorizer, class_weight='balanced', scoring="f1_macro"):
        classifier = LinearSVC()
        # classifier = SVC(kernel="linear")
        # params = {'C': [1, 10, 100, 1000], 'class_weight': [{0: w, -1: w} for w in [2, 3, 4, 5, 6]]}
        params = {'C': [1, 10, 100, 1000], 'class_weight': class_weight}
        # params = {'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001]}

        self.classifier = GridSearchCV(classifier, params, verbose=3, n_jobs=config.n_jobs, scoring=scoring)
        self.vectorizer = vectorizer
        self.select = SelectKBest(chi2, k=2500)

    def features(self, x, y=None, train=False):
        if train:
            logger.debug("select features")
            # print self.vectorizer.transform(x)
            x = self.select.fit_transform(self.vectorizer.transform(x), y)
            logger.debug("select feature done")
        else:
            x = self.select.transform(self.vectorizer.transform(x))
        return x

    def fit(self, x, y):
        self.classifier.fit(self.features(x, y, train=True), y)
        logger.info("Best params:%s,\nBest Score:%s", self.classifier.best_params_, self.classifier.best_score_)

    def predict(self, x):
        return self.classifier.predict(self.features(x))

    def predict_proba(self, x):
        return self.classifier.predict_proba(self.features(x))

    def score(self, x, y):
        return self.classifier.score(self.features(x), y)

    def get_f1_score(self, x, y):
        return f1_score(y, self.predict(x), average='macro')
