from metafeatures.metafeatures import Metafeatures
import joblib
import json
import os
import pandas as pd


DATA_PATH = './data/'


def get_dataset_metafeatures(recommendation_request):
    dataset = pd.read_csv(recommendation_request.dataset_url)
    X = dataset.drop(recommendation_request.target_feature, axis=1)
    y = dataset[recommendation_request.target_feature]

    metafeatures = Metafeatures(X, y, task=recommendation_request.task)
    return metafeatures.compute()


def build_metafeatures_df(metafeatures, task):
    metafeatures_df = pd.DataFrame()
    metafeatures = {'test__'+k: v for k, v in metafeatures.items()}

    pipelines_path = '{}kpis/{}/'.format(DATA_PATH, task)

    for ref_pipeline_filename in os.listdir(pipelines_path):
        pipeline_id = ref_pipeline_filename.split('.')[0]
        ref_pipeline_path = '{}/{}'.format(pipelines_path,
                                           ref_pipeline_filename)
        ref_metafeatures = json.load(open(ref_pipeline_path, 'r'))
        ref_metafeatures = {'ref__'+k: v for k, v in ref_metafeatures.items()}
        metafeatures.update(ref_metafeatures)

        metafeatures_df = metafeatures_df.append(
            pd.Series(metafeatures, name=pipeline_id))
    return metafeatures_df


def get_top_pipeline(metafeatures_df, task, metric='balanced_accuracy'):
    metric = 'balanced_accuracy' if task == 'classification' else 'neg_mean_absolute_error'

    model_path = '{}models/{}_model.joblib'.format(DATA_PATH, metric)
    model = joblib.load(model_path)

    metafeatures_df['predicted_performance'] = model.predict(metafeatures_df)
    # TODO: for regression (MAE, MSE) will need to get idxmin
    top_pipeline_id = metafeatures_df['predicted_performance'].idxmax()
    top_pipeline_path = '{}pipelines/{}/{}.json'.format(
        DATA_PATH, task, top_pipeline_id)
    return json.load(open(top_pipeline_path, 'r'))
