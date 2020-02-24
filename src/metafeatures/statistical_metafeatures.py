from utils import get_stats, get_stats_names, get_numeric_features
from constants import Task
from metafeatures import BaseMetafeaturesComputer


class StatisticalMetafeaturesComputer(BaseMetafeaturesComputer):
    '''
    Calculates statistical/probabilistic metafeatures and returns them as python dictionnary
    '''

    def __init__(self):
        super().__init__(self)

    @staticmethod
    def compute(X, y, task):
        metafeatures = {}
        numeric_features = get_numeric_features(X)
        # TODO: refactor this...
        print(get_numeric_features_means(numeric_features))
        metafeatures.update(get_numeric_features_means(numeric_features))
        metafeatures.update(get_numeric_features_stdevs(numeric_features))
        metafeatures.update(get_numeric_features_variances(numeric_features))
        metafeatures.update(get_numeric_features_minimums(numeric_features))
        metafeatures.update(get_numeric_features_medians(numeric_features))
        metafeatures.update(get_numeric_features_maximums(numeric_features))
        if task == Task.REGRESSION:
            metafeatures.update(
                get_numeric_features_covariances(numeric_features, y))
        return metafeatures


def get_numeric_features_means(numeric_features):
    # numeric_features.iteritems() returns a tuple with column name and pandas series, hence the [1]
    means = [feature[1].mean() for feature in numeric_features.iteritems()]
    return zip(get_stats_names('mean', 'numeric_feautures'), get_stats(means))


def get_numeric_features_stdevs(numeric_features):
    stdevs = [feature[1].std() for feature in numeric_features.iteritems()]
    return zip(get_stats_names('stdev', 'numeric_feautures'), get_stats(stdevs))


def get_numeric_features_variances(numeric_features):
    variances = [feature[1].var()
                 for feature in numeric_features.iteritems()]
    return zip(get_stats_names('var', 'numeric_feautures'), get_stats(variances))


def get_numeric_features_minimums(numeric_features):
    minimums = [feature[1].min()
                for feature in numeric_features.iteritems()]
    return zip(get_stats_names('min', 'numeric_feautures'), get_stats(minimums))


def get_numeric_features_maximums(numeric_features):
    maximums = [feature[1].max()
                for feature in numeric_features.iteritems()]
    return zip(get_stats_names('max', 'numeric_feautures'), get_stats(maximums))


def get_numeric_features_medians(numeric_features):
    medians = [feature[1].median()
               for feature in numeric_features.iteritems()]
    return zip(get_stats_names('median', 'numeric_feautures'), get_stats(medians))


def get_numeric_features_covariances(numeric_features, target):
    covariances = [feature[1].cov(target)
                   for feature in numeric_features.iteritems()]
    return zip(get_stats_names('cov', 'numeric_feautures'), get_stats(covariances))


def get_numeric_features_skews(numeric_features):
    skews = [feature[1].skew() for feature in numeric_features.iteritems()]
    return zip(get_stats_names('skew', 'numeric_features'), get_stats(skews))


def get_numeric_features_kurtosis(numeric_features):
    kurtosis = [feature[1].kurtosis()
                for feature in numeric_features.iteritems()]
    return zip(get_stats_names('kurtosis', 'numeric_features'), get_stats(kurtosis))

# TODO: add quartile1 and quartile3 stats
