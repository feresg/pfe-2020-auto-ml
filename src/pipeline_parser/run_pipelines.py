import json
import os
import pandas as pd

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.datasets import fetch_openml

from tqdm import tqdm

from pipeline_parser import PipelineParser

import warnings
warnings.filterwarnings('ignore')


N_FOLDS = 4
dataset_pipeline_perf_filepath = 'dataset_pipeline_perf.csv'


def run_kfold_on_pipeline(X, y, pipeline):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)
    scores = cross_validate(pipeline, X, y, cv=cv)
    acc = sum(scores['test_score'])/N_FOLDS
    return acc


with open('sampled_models.json', 'r') as f:
    pipelines = json.load(f)

with open('sampled_pipelines.json', 'r') as f:
    pipelines += json.load(f)

classification_datasets = pd.read_csv('datasets.csv')
perfs_df = pd.read_csv('dataset_pipeline_perf_.csv', index_col=0)

datasets_loop = tqdm(classification_datasets.iterrows(),
                     total=classification_datasets.shape[0])

for index, row in datasets_loop:
    dataset_name = row['name']
    datasets_loop.set_description(desc=dataset_name)
    dataset_version = row['version']
    try:
        data, target = fetch_openml(
            name=dataset_name, version=dataset_version, return_X_y=True)
    except ValueError:
        data, target = fetch_openml(
            name=dataset_name, version=dataset_version, as_frame=True, return_X_y=True)

    test_accuracies = {}

    pipelines_loop = tqdm(pipelines)

    for p in pipelines_loop:
        pipelines_loop.set_description(desc=p['id'])

        # TODO: bug (slow models)
        if p['id'] in ['pca_libsvm_svc_14', 'pca_libsvm_svc_4']:
            continue
        pipelines_loop.set_description(desc=p['id'])
        try:
            pipeline = PipelineParser(p).generate_pipeline()
            acc = run_kfold_on_pipeline(data, target, pipeline)
            test_accuracies[p['id']] = acc
        except Exception as e:
            print(str(e))
            continue

    pipeline_performances = pd.DataFrame(
        test_accuracies, index=[dataset_name], columns=perfs_df.columns)
    hdr = False if os.path.isfile(dataset_pipeline_perf_filepath) else True
    pipeline_performances.to_csv(
        dataset_pipeline_perf_filepath, mode='a', header=hdr)
