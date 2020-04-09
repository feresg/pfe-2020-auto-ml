from base_metafeatures import BaseMetafeaturesComputer
from constants import Task
from utils import get_X_y_preprocessed, get_stats, get_stats_names


class CorrelationMetafeatures(BaseMetafeaturesComputer):

    def __init__(self):
        super().__init__(self)

    @staticmethod
    def compute(X, y, task):
        X_preprocessed, y = get_X_y_preprocessed(X, y)
        metafeatures = {}

        if task == Task.REGRESSION:
            metafeatures.update(get_features_correlations(X_preprocessed, y))

        return metafeatures


def get_features_correlations(X, y):
    # numeric_features.iteritems() returns a tuple with column name and pandas series, hence the [1]
    corrs = X.corrwith(y)
    return zip(get_stats_names('correlation', 'features_target'), get_stats(corrs))
