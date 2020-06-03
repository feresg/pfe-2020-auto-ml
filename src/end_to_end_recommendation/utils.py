from metafeatures.metafeatures import Metafeatures
# from h2o import H2OFrame, import_mojo
import h2o
import json
import os
import pandas as pd


h2o.init()

DATA_PATH = '../../data/'


def load_all_models():
    acc_model = h2o.import_mojo(
        '{}models/{}_h2o_model.zip'.format(DATA_PATH, 'accuracy'))
    balaned_acc_model = h2o.import_mojo(
        '{}models/{}_h2o_model.zip'.format(DATA_PATH, 'balanced_accuracy'))

    return {
        'accuracy': acc_model,
        'balanced_accuracy': balaned_acc_model
    }


MODELS = load_all_models()


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
    all_pipelines = filter(lambda f: not f.startswith(
        '.'), os.listdir(pipelines_path))
    for ref_pipeline_filename in all_pipelines:
        pipeline_id = ref_pipeline_filename.split('.')[0]
        ref_pipeline_path = '{}/{}'.format(pipelines_path,
                                           ref_pipeline_filename)
        ref_metafeatures = json.load(open(ref_pipeline_path, 'r'))
        ref_metafeatures = {'ref__'+k: v for k, v in ref_metafeatures.items()}
        metafeatures.update(ref_metafeatures)

        metafeatures_df = metafeatures_df.append(
            pd.Series(metafeatures, name=pipeline_id))
    # print(metafeatures_df)
    return metafeatures_df


def get_top_pipeline(metafeatures_df, task, metric='balanced_accuracy', n=5):
    # metric = 'balanced_accuracy' if task == 'classification' else 'neg_mean_absolute_error'
    model = MODELS[metric]

    test_df = h2o.H2OFrame(metafeatures_df)
    test_df = test_df.asnumeric()

    predictions = model.predict(test_df)
    predictions = predictions.as_data_frame(use_pandas=True)
    predictions.index = metafeatures_df.index

    top_n_pipelines = predictions['predict'].nlargest(
        n=10).index.values.tolist()

    # TODO: for regression (MAE, MSE) will need to get idxmin
    top_pipeline_id = predictions['predict'].idxmax()
    top_pipeline_path = '{}pipelines/{}/{}.json'.format(
        DATA_PATH, task, top_pipeline_id)
    return top_pipeline_id, json.load(open(top_pipeline_path, 'r')), top_n_pipelines
