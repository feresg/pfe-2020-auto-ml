from .utils import get_stats, get_stats_names, get_numeric_features, get_X_y_preprocessed
from .constants import Task, empty_dataframe_stats
from .base_metafeatures import BaseMetafeaturesComputer

import numpy as np
from sklearn.decomposition import PCA


class StatisticalMetafeaturesComputer(BaseMetafeaturesComputer):
    '''
    Calculates statistical/probabilistic metafeatures and returns them as python dictionnary
    '''

    def __init__(self):
        super().__init__(self)

    @staticmethod
    def compute(X, y, task):

        X_preprocessed, y = get_X_y_preprocessed(X, y)
        numeric_features = get_numeric_features(X_preprocessed)

        metafeatures = {}

        # TODO: refactor this...
        metafeatures.update(get_numeric_features_means(numeric_features))
        metafeatures.update(get_numeric_features_stdevs(numeric_features))
        metafeatures.update(get_numeric_features_variances(numeric_features))
        metafeatures.update(get_numeric_features_minimums(numeric_features))
        metafeatures.update(get_numeric_features_medians(numeric_features))
        metafeatures.update(get_numeric_features_maximums(numeric_features))
        metafeatures.update(get_pca_variance_percentages(numeric_features))

        if task == Task.REGRESSION:
            metafeatures.update(
                get_numeric_features_covariances(numeric_features, y))

        return metafeatures


def get_numeric_features_means(numeric_features):
    if not numeric_features.empty:
        means = [feature.mean() for _, feature in numeric_features.iteritems()]
        return zip(get_stats_names('mean', 'numeric_features'), get_stats(means))
    else:
        return zip(get_stats_names('mean', 'numeric_features'), empty_dataframe_stats)


def get_numeric_features_stdevs(numeric_features):
    if not numeric_features.empty:
        stdevs = [feature.std() for _, feature in numeric_features.iteritems()]
        return zip(get_stats_names('stdev', 'numeric_features'), get_stats(stdevs))
    else:
        return zip(get_stats_names('stdev', 'numeric_features'), empty_dataframe_stats)


def get_numeric_features_variances(numeric_features):
    if not numeric_features.empty:
        variances = [feature.var()
                     for _, feature in numeric_features.iteritems()]
        return zip(get_stats_names('var', 'numeric_features'), get_stats(variances))
    else:
        return zip(get_stats_names('var', 'numeric_features'), empty_dataframe_stats)


def get_numeric_features_minimums(numeric_features):
    if not numeric_features.empty:
        minimums = [feature.min()
                    for _, feature in numeric_features.iteritems()]
        return zip(get_stats_names('min', 'numeric_features'), get_stats(minimums))
    else:
        return zip(get_stats_names('min', 'numeric_features'), empty_dataframe_stats)


def get_numeric_features_maximums(numeric_features):
    if not numeric_features.empty:
        maximums = [feature.max()
                    for _, feature in numeric_features.iteritems()]
        return zip(get_stats_names('max', 'numeric_features'), get_stats(maximums))
    else:
        return zip(get_stats_names('max', 'numeric_features'), empty_dataframe_stats)


def get_numeric_features_medians(numeric_features):
    if not numeric_features.empty:
        medians = [feature.median()
                   for _, feature in numeric_features.iteritems()]
        return zip(get_stats_names('median', 'numeric_features'), get_stats(medians))
    else:
        return zip(get_stats_names('median', 'numeric_features'), empty_dataframe_stats)


def get_numeric_features_covariances(numeric_features, target):
    if not numeric_features.empty:
        covariances = [feature.cov(target)
                       for _, feature in numeric_features.iteritems()]
        return zip(get_stats_names('cov', 'numeric_features'), get_stats(covariances))
    else:
        return zip(get_stats_names('cov', 'numeric_features'), empty_dataframe_stats)


def get_numeric_features_skews(numeric_features):
    if not numeric_features.empty:
        skews = [feature.skew() for _, feature in numeric_features.iteritems()]
        return zip(get_stats_names('skew', 'numeric_features'), get_stats(skews))
    else:
        return zip(get_stats_names('skew', 'numeric_features'), empty_dataframe_stats)


def get_numeric_features_kurtosis(numeric_features):
    if not numeric_features.empty:
        kurtosis = [feature.kurtosis()
                    for _, feature in numeric_features.iteritems()]
        return zip(get_stats_names('kurtosis', 'numeric_features'), get_stats(kurtosis))
    else:
        return zip(get_stats_names('kurtosis', 'numeric_features'), empty_dataframe_stats)


def get_pca_variance_percentages(numeric_features):
    pred_pca_names = ['pca_1', 'pca_2', 'pca_3']
    if not numeric_features.empty:

        num_components = min(3, numeric_features.shape[1])
        try:
            pca_data = PCA(n_components=num_components)
            pca_data.fit_transform(numeric_features.values)
            pred_pca = pca_data.explained_variance_ratio_
        except Exception:
            pred_pca = [np.nan, np.nan, np.nan]
    else:
        pred_pca = [np.nan, np.nan, np.nan]
    return zip(pred_pca_names, pred_pca)
