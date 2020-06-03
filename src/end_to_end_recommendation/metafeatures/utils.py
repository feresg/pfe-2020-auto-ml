from functools import wraps
from time import time

import numpy as np
import pandas as pd

from .constants import empty_dataframe_stats

import logging
logging.basicConfig(filename='execution_time.log', filemode='a',
                    format='%(asctime)s - %(message)s', level=logging.INFO)


def fix_X_column_dtypes(X):
    obj_bool_columns = X.select_dtypes(['object', 'bool']).columns
    X[obj_bool_columns] = X[obj_bool_columns].apply(
        lambda x: x.astype('category'))
    cat_columns = X.select_dtypes(['category']).columns
    X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)
    return X


def get_X_y_preprocessed(X, y):
    target = y.name
    X = fix_X_column_dtypes(X)
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


# def get_stats(data):
#     data = np.asarray(data, dtype=np.float32)
#     data = data[~np.isnan(data)]
#     if data.size == 0:
#         return empty_dataframe_stats
#     dist_mean = np.mean(data)
#     dist_stdev = np.std(data)

#     if no_quartiles:
#         dist_min, dist_median, dist_max = np.percentile(
#             data, [0, 50, 100])
#         return (dist_mean, dist_stdev, dist_min, dist_median, dist_max)

#     dist_min, dist_quartile1, dist_median, dist_quartile3, dist_max = np.percentile(
#         data, [0, 25, 50, 75, 100])
#     return (dist_mean, dist_stdev, dist_min, dist_quartile1, dist_median, dist_quartile3, dist_max)

def get_stats_names(stats, measure, datatype):
    metafeature_names = [x + '_' + measure + '_' + datatype for x in stats]
    return metafeature_names


def get_stats(data, stats):
    data = np.asarray(data, dtype=np.float32)
    data = data[~np.isnan(data)]
    if data.size == 0:
        return [np.nan] * len(stats)

    res = []

    stat_calculator = {
        'mean': np.mean,
        'stdev': np.mean,
        'min': np.min,
        'max': np.max,
        'median': np.median,
        'q1': lambda x: np.quantile(x, .25),
        'q3': lambda x: np.quantile(x, .75)
    }

    for stat in stats:
        res.append(stat_calculator[stat](data))
    return res


def timeit(f):
    @wraps(f)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return f(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            logging.info(
                f"{f.__name__} execution time: {end_ if end_ > 0 else 0} ms")
    return _time_it
