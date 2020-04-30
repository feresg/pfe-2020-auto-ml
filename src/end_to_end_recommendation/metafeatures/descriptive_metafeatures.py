from scipy.stats import entropy
import numpy as np

from .base_metafeatures import BaseMetafeaturesComputer
from .constants import Task
from .utils import get_numeric_features, get_categorical_features, get_stats, get_stats_names


class DescriptiveMetafeatures(BaseMetafeaturesComputer):
    def __init__(self):
        super().__init__(self)

    @staticmethod
    def compute(X, y, task):
        numeric_features = get_numeric_features(X)
        categorical_features = get_categorical_features(X)

        metafeatures = {}

        metafeatures['nb_instances'] = X.shape[0]
        metafeatures['log_nb_instances'] = np.log(metafeatures['nb_instances'])

        metafeatures['nb_features'] = X.shape[1]

        metafeatures['dataset_dimensionality'] = X.shape[0] / X.shape[1]
        metafeatures['log_dataset_dimensionality'] = np.log(
            metafeatures['dataset_dimensionality'])
        metafeatures['dataset_ratio'] = X.shape[1] / X.shape[0]

        metafeatures['nb_missing_vals'] = X.isnull().sum().sum()
        metafeatures['ratio_features_with_missing_vals'] = X.isnull().any(
            axis=0).sum() / X.shape[0]
        metafeatures['ratio_instances_with_missing_vals'] = X.isnull().any(
            axis=1).sum() / X.shape[0]

        metafeatures['nb_numeric_features'] = numeric_features.shape[1]
        metafeatures['nb_categorical_features'] = categorical_features.shape[1]
        metafeatures['ratio_numeric_categorical_features'] = 0.0 if metafeatures[
            'nb_categorical_features'] == 0 else metafeatures['nb_numeric_features'] / metafeatures['nb_categorical_features']
        metafeatures['ratio_categorical_numeric_features'] = 0.0 if metafeatures[
            'nb_numeric_features'] == 0 else metafeatures['nb_categorical_features'] / metafeatures['nb_numeric_features']

        if task == Task.CLASSIFICATION:
            metafeatures['nb_classes'] = y.value_counts().shape[0]
            metafeatures['class_entropy'] = entropy(y.value_counts())

            metafeatures.update(get_target_class_probabilities_stats(y))

        elif task == Task.REGRESSION:
            metafeatures.update(get_regression_target_stats(y))
        return metafeatures


def get_regression_target_stats(y):
    stats = ['mean', 'stdev', 'min', 'q1', 'median', 'q3', 'max']
    regression_stats_names = [stat + '__target' for stat in stats]
    return zip(regression_stats_names, get_stats(y))


def get_target_class_probabilities_stats(y):
    stats = ['mean', 'stdev', 'minority', 'median', 'majority']
    target_class_probabilities_names = [
        stat + '_target_class_ratio' for stat in stats]

    target_class_probabilities = y.value_counts().values / y.shape[0]
    return zip(target_class_probabilities_names, get_stats(target_class_probabilities, no_quartiles=True))
