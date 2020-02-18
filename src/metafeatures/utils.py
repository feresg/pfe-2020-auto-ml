from functools import wraps
from time import time

import numpy as np
import pandas as pd


def get_stats(data):
    dist_mean = np.mean(data)
    dist_stdev = np.std(data)
    dist_min, dist_quartile1, dist_median, dist_quartile3, dist_max = np.percentile(
        data, [0, 25, 50, 75, 100])
    return (dist_mean, dist_stdev, dist_min, dist_quartile1, dist_median, dist_quartile3, dist_max)


def get_stats_names(measure, datatype):
    stats = ['mean', 'stdev', 'min', 'q1', 'median', 'q3', 'max']
    def generate_metafeature_name(
        x): return x + '_' + measure + '__' + datatype
    return [generate_metafeature_name(stat) for stat in stats]


def get_numeric_features(df):
    numeric_feature_names = [
        feature for feature in df.columns if pd.api.types.is_numeric_dtype(df[feature])]
    return df[numeric_feature_names]


def get_categorical_features(df):
    categorical_feature_names = [
        feature for feature in df.columns if pd.api.types.is_categorical_dtype(df[feature])]
    return df[categorical_feature_names]


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
