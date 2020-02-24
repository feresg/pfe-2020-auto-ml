import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from constants import Task
from metafeatures import BaseMetafeaturesComputer
'''
    Landmarking metafeatures based on training results on machine learning models with low complexity
    (Training complexity lower than O(n*log(n)*p)) where:
        n: number of observations
        p: number of features
'''


class LandmarkingMetafeatures(BaseMetafeaturesComputer):

    def __init__(self):
        super().__init__(self)

    @staticmethod
    def compute(X, y, task):
        metafeatures = {}
        if task == Task.CLASSIFICATION:
            metafeatures['knn_5_accuracy'] = get_cross_val_score(
                KNeighborsClassifier(n_neighbors=5), X, y)
            metafeatures['gaussian_naive_bayes_accuracy'] = get_cross_val_score(
                GaussianNB(), X, y)
            # TODO: add more models accuracies
        elif task == Task.REGRESSION:
            # TODO: get mean losses on simple regression models
            pass
        return metafeatures


# classifiers = dict(
#     knn_5=KNeighborsClassifier(n_neighbors=5),
#     gaussian_naive_bayes=GaussianNB(),
# )


def get_cross_val_score(model, X, y, n_folds=5):
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True)
    scores = cross_validate(model, X.values, y.values, cv=cv)
    return np.mean(scores['test_score'])
