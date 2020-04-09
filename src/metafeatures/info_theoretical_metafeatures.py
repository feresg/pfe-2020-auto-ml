
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
import numpy as np

from utils import get_stats, get_stats_names, get_categorical_features, get_X_y_preprocessed
from constants import Task, empty_dataframe_stats_no_q
from base_metafeatures import BaseMetafeaturesComputer


class InfoTheoreticalMetafeatures(BaseMetafeaturesComputer):

    def __init__(self):
        super().__init__(self)

    @staticmethod
    def compute(X, y, task):

        X_preprocessed, y = get_X_y_preprocessed(X, y)
        categorical_features = get_categorical_features(X_preprocessed)

        metafeatures = {}

        metafeatures.update(
            get_entropies_categorical_features(categorical_features))
        if task == Task.CLASSIFICATION:
            metafeatures.update(
                get_information_gain_categorical_features(categorical_features, y))
            metafeatures.update(
                get_mutual_information_categorical_features(categorical_features, y))
        return metafeatures


def information_gain(feature, target):
    try:
        entropy_before = entropy(feature.value_counts())
        grouped_distrib = feature.groupby(target).value_counts(sort=True).reset_index(
            name='count').pivot_table(index=target.name, columns=feature.name, values='count')
        entropy_after = entropy(grouped_distrib, axis=1)
        entropy_after *= target.value_counts(sort=True, normalize=True)
        return entropy_before - entropy_after.sum()
    except Exception:
        return np.nan


def get_entropies_categorical_features(categorical_features):
    if not categorical_features.empty:
        entropies = [entropy(feature.value_counts())
                     for _, feature in categorical_features.iteritems()]

        return zip(get_stats_names('entropy', 'categorical_features', no_quartiles=True), get_stats(entropies, no_quartiles=True))
    else:
        return zip(get_stats_names('entropy', 'categorical_features', no_quartiles=True), empty_dataframe_stats_no_q)


def get_information_gain_categorical_features(categorical_features, y):
    if not categorical_features.empty:
        information_gains = [information_gain(
            feature, y) for _, feature in categorical_features.iteritems()]
        return zip(get_stats_names('information_gain', 'categorical_features', no_quartiles=True), get_stats(information_gains, no_quartiles=True))
    # temporary fix if no categorical features (needed to keep the same number of metafeatures across different datasets)
    else:
        return zip(get_stats_names('information_gain', 'categorical_features', no_quartiles=True), empty_dataframe_stats_no_q)


def get_mutual_information_categorical_features(categorical_features, y):
    if not categorical_features.empty:
        mutual_informations = [mutual_info_score(
            feature, y) for _, feature in categorical_features.iteritems()]
        return zip(get_stats_names('mutual_informaiton', 'categorical_features', no_quartiles=True), get_stats(mutual_informations, no_quartiles=True))
    # temporary fix if no categorical features (needed to keep the same number of metafeatures across different datasets)
    else:
        return zip(get_stats_names('mutual_informaiton', 'categorical_features', no_quartiles=True), empty_dataframe_stats_no_q)
