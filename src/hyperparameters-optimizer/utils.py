import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold, cross_validate
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler

from .config import classification_config, regression_config


class TimeStopper(tune.Stopper):
    def __init__(self, deadline=3600):
        self._start = time.time()
        self._laps = [self._start]
        self._deadline = deadline

    def __call__(self, trial_id, result):
        return False

    def stop_all(self):
        # TODO: we should stop also if elapsed time + avg time per trial > deadline
        return time.time() - self._start > self._deadline


def train_classification_pipeline(config, pipeline, X, y):
    scoring = ['accuracy', 'balanced_accuracy']
    pipeline = pipeline.set_params(**config)
    cv = KFold(n_splits=2, shuffle=True)
    # print('starting train...')
    scores = cross_validate(
        pipeline, X, y, cv=cv, scoring=scoring)
    # print(scores)
    tune.track.log(
        mean_accuracy=np.mean(scores['test_accuracy']),
        mean_balanced_accuracy=np.mean(scores['test_balanced_accuracy']),
        mean_fit_time=np.mean(scores['fit_time']),
        mean_score_time=np.mean(scores['score_time']),
        # pipeline=pipeline,
        done=True
    )


def train_regression_pipeline(config, pipeline, X, y):
    scoring = ['r2', 'neg_mean_absolute_error',
               'neg_mean_squared_error', 'neg_squared_mean_squared_error']
    pipeline = pipeline.set_params(**config)
    cv = KFold(n_splits=4, shuffle=True)
    # print('starting train...')
    scores = cross_validate(
        pipeline, X, y, cv=cv, scoring=scoring)
    # print(scores)
    tune.track.log(
        mean_r2=np.mean(scores['test_r2']),
        mean_neg_mean_absolute_error=np.mean(
            scores['test_neg_mean_absolute_error']),
        mean_neg_mean_squared_error=np.mean(
            scores['test_neg_mean_squared_error']),
        mean_neg_squared_mean_squared_error=np.mean(
            scores['test_neg_squared_mean_squared_error']),
        mean_fit_time=np.mean(scores['fit_time']),
        mean_score_time=np.mean(scores['score_time']),
        # pipeline=pipeline,
        done=True
    )


def train_pipeline(config, task, pipeline, X, y):
    if task == 'classification':
        return train_classification_pipeline(config, pipeline, X, y)
    return train_regression_pipeline(config, pipeline, X, y)

# TODO: dynamically setup objective (maximize, minimize) based on metric


def build_pipeline_search_space(task, pipeline):
    config = classification_config if task == 'classification' else regression_config
    # pipeline[-1] is the estimator/model
    pipeline_estimator_name = pipeline.steps[-1][0]
    return config[pipeline_estimator_name]


def build_hyperopt_search_algorithm(task, metric, space):
    metric = 'mean_' + metric
    if task == 'classification':
        algo = HyperOptSearch(
            space,
            metric=metric,
            mode="max"
        )
    return algo

# TODO: dynamically setup objective (maximize, minimize) based on metric


def build_hyperopt_scheduler(metric):
    metric = 'mean_' + metric
    scheduler = AsyncHyperBandScheduler(
        metric=metric, mode="max"
    )
    return scheduler
