'''
inputs:
    pipeline_id
    dataset path
    target feature name
    task type (classification, regression)
    metric to optimize
    time (in seconds)

returns:
    pipeline (as json) with new params
    OR
    best params dictionnary

'''

import pandas as pd
import joblib
import ray
from ray import tune

from .utils import train_pipeline, build_pipeline_search_space, build_hyperopt_scheduler, build_hyperopt_search_algorithm, TimeStopper

ray.init()

DATA_PATH = '../../data/'


def run_hyperparameter_optimization_experiment(
    pipeline_id,
    dataset_url,
    target_feature,
    task,
    metric,
    deadline
):
    # step 1: load dataset, create X and y
    dataset = pd.read_csv(dataset_url)
    X = dataset.drop(target_feature, axis=1)
    y = dataset[target_feature]

    # step 2: load pipeline
    pipeline = joblib.load(
        '{}pipelines_binary/{}/{}.joblib'.format(DATA_PATH, task, pipeline_id))

    space = build_pipeline_search_space(task, pipeline)

    search_algorithm = build_hyperopt_search_algorithm(task, metric, space)
    search_scheduler = build_hyperopt_scheduler(metric)

    experiment = tune.run(
        lambda x: train_pipeline(x, task, pipeline, X, y),
        scheduler=search_scheduler,
        search_alg=search_algorithm,
        num_samples=100,  # large enough to not cause early stopping
        stop=TimeStopper(deadline)
    )

    # TODO: access experiment to get best results
    best_params = experiment.get_best_config(metric=metric)
    return best_params
