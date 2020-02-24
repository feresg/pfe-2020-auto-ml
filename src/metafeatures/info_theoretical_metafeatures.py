
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

from utils import get_stats, get_stats_names, get_categorical_features
from constants import Task
from metafeatures import BaseMetafeaturesComputer


class InfoTheoreticalMetafeatures(BaseMetafeaturesComputer):

    def __init__(self):
        super().__init(self)

    @staticmethod
    def compute(X, y, task):
        metafeatures = {}
        categorical_features = get_categorical_features(X)
        if task == Task.CLASSIFICATION:
            metafeatures['class_entropy'] = entropy(y.value_counts())
            metafeatures.update(
                get_information_gain_categorical_features(categorical_features, y))
            metafeatures.update(get_mutual_information_categorical_features(categorical_features, y)))
        return metafeatures


def information_gain(feature, target):
    entropy_before=entropy(feature.value_counts())
    grouped_distrib=feature.groupby(target).value_counts(sort = True).reset_index(
        name = 'count').pivot_table(index = target.name, columns = feature.name, values = 'count')
    entropy_after=entropy(grouped_distrib, axis = 1)
    entropy_after *= target.value_counts(sort = True, normalize = True)
    return entropy_before - entropy_after.sum()


def get_categorical_features_entropies(categorical_features):
    entropies=[entropy(feature[1])
                 for feature in categorical_features.iteritems()]
    return zip(get_stats_names('entropy', 'categorical_features'), get_stats(entropies))


def get_information_gain_categorical_features(categorical_features, y):
    information_gains = [information_gain(
        feature[1], y) for feature in categorical_features.iteritems()]
    return zip(get_stats_names('information_gain', 'categorical_features'), get_stats(information_gains))


def get_mutual_information_categorical_features(categorical_features, y):
    mutual_informations = [mutual_info_score(
        feature[1], y) for feature in categorical_features.iteritems()]
    return zip(get_stats_names('mutual_informaiton', 'categorical_features'), get_stats(mutual_informations))
