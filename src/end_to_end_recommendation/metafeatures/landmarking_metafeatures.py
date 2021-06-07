import numpy as np
import sklearn
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from .constants import Task
from .utils import get_X_y_preprocessed
from .base_metafeatures import BaseMetafeaturesComputer

import logging
# logging.basicConfig(filename='landmarking.log', filemode='a',
#                     format='%(asctime)s - %(message)s', level=logging.INFO)


'''
    Landmarking metafeatures based on training results on machine learning models with low complexity
    (Training complexity lower than O(n*log(n)*p)) where:
        n: number of observations
        p: number of features
'''

seed = sklearn.utils.check_random_state(42)


class LandmarkingMetafeatures(BaseMetafeaturesComputer):

    def __init__(self):
        super().__init__(self)

    @staticmethod
    def compute(X, y, task):

        X_preprocessed, y = get_X_y_preprocessed(X, y)

        metafeatures = {}
        if task == Task.CLASSIFICATION:
            metafeatures.update(get_classification_scores(X_preprocessed, y))
        elif task == Task.REGRESSION:
            metafeatures.update(get_regression_scores(X_preprocessed, y))
        return metafeatures


classifiers = dict(
    # knn_5=KNeighborsClassifier(n_neighbors=5),
    gaussian_nb=GaussianNB(),
    lda=LinearDiscriminantAnalysis(),
    decision_tree_1=DecisionTreeClassifier(
        max_depth=1, criterion='entropy', random_state=seed),
    decision_tree_2=DecisionTreeClassifier(
        max_depth=2, criterion='entropy', random_state=seed),
    extra_trees_2_10=ExtraTreesClassifier(
        max_depth=2, n_estimators=10, random_state=seed)
)

regressors = dict(
    # knn_5=KNeighborsRegressor(n_neighbors=5),
    decision_tree_1=DecisionTreeRegressor(max_depth=1, random_state=seed),
    decision_tree_2=DecisionTreeRegressor(max_depth=2, random_state=seed),
    random_tree_2=DecisionTreeRegressor(
        max_depth=2, splitter='random', random_state=seed)
    # extra_trees_2_10=ExtraTreesRegressor(
    #     max_depth=2, n_estimators=10, random_state=seed)
)


def get_classification_scores(X, y, n_folds=5):
    # TODO: add more classification metrics
    scoring = ['accuracy', 'balanced_accuracy']
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    landmarking_scores = {}
    for classifier_name, classifier in classifiers.items():
        scores = cross_validate(
            classifier, X.values, y.values, cv=cv, scoring=scoring)
        landmarking_scores[classifier_name +
                           '_accuracy'] = np.nanmean(scores['test_accuracy'])
        landmarking_scores[classifier_name +
                           '_balanced_accuracy'] = np.nanmean(scores['test_balanced_accuracy'])

        avg_time = np.nanmean(scores['fit_time']) + \
            np.nanmean(scores['score_time'])
        logging.info('{}: {} s'.format(classifier_name, avg_time))
    return landmarking_scores


def get_regression_scores(X, y, n_folds=4):
    scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    landmarking_scores = {}
    for regressor_name, regressor in regressors.items():
        scores = cross_validate(
            regressor, X.values, y.values, cv=cv, scoring=scoring)
        landmarking_scores[regressor_name +
                           '_mse'] = -1.0 * np.nanmean(scores['test_neg_mean_squared_error'])
        landmarking_scores[regressor_name +
                           '_mae'] = -1.0 * np.nanmean(scores['test_neg_mean_absolute_error'])
        landmarking_scores[regressor_name +
                           '_r2'] = np.nanmean(scores['test_r2'])

        avg_time = np.nanmean(scores['fit_time']) + \
            np.nanmean(scores['score_time'])
        logging.info('{}: {} s'.format(regressor_name, avg_time))
    return landmarking_scores
