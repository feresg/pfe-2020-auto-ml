
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
import numpy as np

from .utils import get_stats, get_stats_names, get_categorical_features, get_X_y_preprocessed, timeit
from .constants import Task
from .base_metafeatures import BaseMetafeaturesComputer


class InfoTheoreticalMetafeatures(BaseMetafeaturesComputer):

    def __init__(self):
        super().__init__(self)

    @staticmethod
    @timeit
    def compute(X, y, task):

        X_preprocessed, y = get_X_y_preprocessed(X, y)
        categorical_features = get_categorical_features(X_preprocessed)

        metafeatures = {}

        metafeatures.update(
            get_entropies_categorical_features(categorical_features))
        if task == Task.CLASSIFICATION:
            # metafeatures.update(
            #     get_information_gain_categorical_features(categorical_features, y))
            metafeatures.update(
                get_mutual_information_categorical_features(categorical_features, y))
        return metafeatures


def information_gain(feature, target):
    try:
        entropy_before = entropy(feature.value_counts(normalize=True))
        grouped_distrib = feature.groupby(target) \
            .value_counts(normalize=True) \
            .reset_index(name='count') \
            .pivot_table(index=target.name, columns=feature.name, values='count') \
            .fillna(0)
        entropy_after = entropy(grouped_distrib, axis=1)
        entropy_after *= target.value_counts(sort=False, normalize=True)
        return entropy_before - entropy_after.sum()
    except Exception as e:
        return np.nan


@timeit
def get_entropies_categorical_features(categorical_features, stats=['mean', 'stdev']):
    stats_names = get_stats_names(stats, 'entropy', 'categorical_features')
    if not categorical_features.empty:
        entropies = [entropy(feature.value_counts())
                     for _, feature in categorical_features.iteritems()]

        return zip(stats_names, get_stats(entropies, stats))
    else:
        return zip(stats_names, [np.nan, np.nan])


@timeit
def get_information_gain_categorical_features(categorical_features, y, stats=['mean', 'stdev']):
    stats_names = get_stats_names(
        stats, 'information_gain', 'categorical_features')
    if not categorical_features.empty:
        information_gains = [information_gain(
            feature, y) for _, feature in categorical_features.iteritems()]
        return zip(stats_names, get_stats(information_gains, stats))
    # temporary fix if no categorical features (needed to keep the same number of metafeatures across different datasets)
    else:
        return zip(stats_names, [np.nan, np.nan])


@timeit
def get_mutual_information_categorical_features(categorical_features, y, stats=['mean', 'stdev']):
    stats_names = get_stats_names(
        stats, 'mutual_informaiton', 'categorical_features')
    if not categorical_features.empty:
        mutual_informations = [mutual_info_score(
            feature, y) for _, feature in categorical_features.iteritems()]
        return zip(stats_names, get_stats(mutual_informations, stats))
    # temporary fix if no categorical features (needed to keep the same number of metafeatures across different datasets)
    else:
        return zip(stats_names, [np.nan, np.nan])
