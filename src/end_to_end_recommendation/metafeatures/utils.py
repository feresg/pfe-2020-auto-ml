from functools import wraps
from time import time

import numpy as np
import pandas as pd

from .constants import empty_dataframe_stats


def get_X_y_preprocessed(X, y):
    target = y.name
    df = pd.concat([X, y], axis=1)
    df = df.dropna(how='all', axis=1)  # drop columns with all nan values
    df = df.dropna()  # drop rows containing null values
    return df.drop(target, axis=1), df[target]


def get_numeric_features(X):
    numeric_feature_names = [
        feature for feature in X.columns if pd.api.types.is_numeric_dtype(X[feature])]
    return X[numeric_feature_names]


def get_categorical_features(X):
    categorical_feature_names = [
        feature for feature in X.columns if pd.api.types.is_categorical_dtype(X[feature])]
    return X[categorical_feature_names]


def get_stats(data, no_quartiles=False):
    data = np.asarray(data, dtype=np.float32)
    data = data[~np.isnan(data)]
    if data.size == 0:
        return empty_dataframe_stats
    dist_mean = np.mean(data)
    dist_stdev = np.std(data)

    if no_quartiles:
        dist_min, dist_median, dist_max = np.percentile(
            data, [0, 50, 100])
        return (dist_mean, dist_stdev, dist_min, dist_median, dist_max)

    dist_min, dist_quartile1, dist_median, dist_quartile3, dist_max = np.percentile(
        data, [0, 25, 50, 75, 100])
    return (dist_mean, dist_stdev, dist_min, dist_quartile1, dist_median, dist_quartile3, dist_max)


def get_stats_names(measure, datatype, no_quartiles=False):
    stats = ['mean', 'stdev', 'min', 'median', 'max'] if no_quartiles else [
        'mean', 'stdev', 'min', 'q1', 'median', 'q3', 'max']

    metafeature_names = [x + '_' + measure + '_' + datatype for x in stats]
    return metafeature_names


def timeit(f):
    @wraps(f)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return f(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            print(f"Total execution time: {end_ if end_ > 0 else 0} ms")
    return _time_it
