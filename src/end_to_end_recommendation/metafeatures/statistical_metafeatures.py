from .utils import get_stats, get_stats_names, get_numeric_features, get_X_y_preprocessed, timeit
from .constants import Task
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
    @timeit
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
        metafeatures.update(get_numeric_features_skews(numeric_features))
        metafeatures.update(get_numeric_features_kurtosis(numeric_features))
        metafeatures.update(get_pca_variance_percentages(numeric_features))

        if task == Task.REGRESSION:
            metafeatures.update(
                get_numeric_features_covariances(numeric_features, y))

        return metafeatures


@timeit
def get_numeric_features_means(numeric_features, stats=['mean', 'stdev']):
    stats_names = get_stats_names(stats, 'mean', 'numeric_features')
    if not numeric_features.empty:
        means = [feature.mean() for _, feature in numeric_features.iteritems()]
        return zip(stats_names, get_stats(means, stats))
    else:
        return zip(stats_names, [np.nan, np.nan])


@timeit
def get_numeric_features_stdevs(numeric_features, stats=['mean', 'stdev']):
    stats_names = get_stats_names(stats, 'stdev', 'numeric_features')
    if not numeric_features.empty:
        stdevs = [feature.std() for _, feature in numeric_features.iteritems()]
        return zip(stats_names, get_stats(stdevs, stats))
    else:
        return zip(stats_names, [np.nan, np.nan])


@timeit
def get_numeric_features_variances(numeric_features, stats=['mean', 'stdev']):
    stats_names = get_stats_names(stats, 'var', 'numeric_features')
    if not numeric_features.empty:
        variances = [feature.var()
                     for _, feature in numeric_features.iteritems()]
        return zip(stats_names, get_stats(variances, stats))
    else:
        return zip(stats_names, [np.nan, np.nan])


@timeit
def get_numeric_features_minimums(numeric_features, stats=['mean', 'stdev']):
    stats_names = get_stats_names(stats, 'min', 'numeric_features')
    if not numeric_features.empty:
        minimums = [feature.min()
                    for _, feature in numeric_features.iteritems()]
        return zip(stats_names, get_stats(minimums, stats))
    else:
        return zip(stats_names, [np.nan, np.nan])


@timeit
def get_numeric_features_maximums(numeric_features, stats=['mean', 'stdev']):
    stats_names = get_stats_names(stats, 'max', 'numeric_features')
    if not numeric_features.empty:
        maximums = [feature.max()
                    for _, feature in numeric_features.iteritems()]
        return zip(stats_names, get_stats(maximums, stats))
    else:
        return zip(stats_names, [np.nan, np.nan])


@timeit
def get_numeric_features_medians(numeric_features, stats=['mean', 'stdev']):
    stats_names = get_stats_names(stats, 'median', 'numeric_features')
    if not numeric_features.empty:
        medians = [feature.median()
                   for _, feature in numeric_features.iteritems()]
        return zip(stats_names, get_stats(medians, stats))
    else:
        return zip(stats_names, [np.nan, np.nan])


@timeit
def get_numeric_features_covariances(numeric_features, target, stats=['mean', 'stdev']):
    stats_names = get_stats_names(stats, 'cov', 'numeric_features')
    if not numeric_features.empty:
        covariances = [feature.cov(target)
                       for _, feature in numeric_features.iteritems()]
        return zip(stats_names, get_stats(covariances, stats))
    else:
        return zip(stats_names, [np.nan, np.nan])


@timeit
def get_numeric_features_skews(numeric_features, stats=['mean', 'stdev']):
    stats_names = get_stats_names(stats, 'skew', 'numeric_features')
    if not numeric_features.empty:
        skews = [feature.skew() for _, feature in numeric_features.iteritems()]
        return zip(stats_names, get_stats(skews, stats))
    else:
        return zip(stats_names, [np.nan, np.nan])


@timeit
def get_numeric_features_kurtosis(numeric_features, stats=['mean', 'stdev']):
    stats_names = get_stats_names(stats, 'kurtosis', 'numeric_features')
    if not numeric_features.empty:
        kurtosis = [feature.kurtosis()
                    for _, feature in numeric_features.iteritems()]
        return zip(stats_names, get_stats(kurtosis, stats))
    else:
        return zip(stats_names, [np.nan, np.nan])


@timeit
def get_pca_variance_percentages(numeric_features):
    pred_pca_names = ['pca_1', 'pca_2', 'pca_3']
    if not numeric_features.empty:

        num_components = min(3, numeric_features.shape[1])
        try:
            pca_data = PCA(n_components=num_components)
            pca_data.fit_transform(numeric_features.values)
            pred_pca_ = pca_data.explained_variance_ratio_
            # padding in case num components < 3
            pred_pca = np.pad(pred_pca_, (0, 3 - len(pred_pca_)),
                              mode='constant', constant_values=np.nan)
        except Exception:
            pred_pca = [np.nan, np.nan, np.nan]
    else:
        pred_pca = [np.nan, np.nan, np.nan]
    return zip(pred_pca_names, pred_pca)
