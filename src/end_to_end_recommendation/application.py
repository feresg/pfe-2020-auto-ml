from utils import get_dataset_metafeatures, build_metafeatures_df, get_top_pipeline
import time

from flask import Flask, request, jsonify
from flask_cors import cross_origin

app = Flask(__name__)


class RecommendationRequest:

    def __init__(self, dataset_url, target_feature, task, metric):
        self.dataset_url = dataset_url
        self.target_feature = target_feature
        self.task = task
        self.metric = metric

# TODO: return http error status if task is not in [classification, regression], or invalid dataset url or if target_feature not in dataset
@app.route('/recommend', methods=['GET'])
@cross_origin()
def recommend_pipeline():
    # request data contains dataset path/url, target feature name and task type (classification/regression)
    recommendation_request = RecommendationRequest(
        dataset_url=request.json['dataset_url'],
        target_feature=request.json['target_feature'],
        task=request.json['task'],
        metric=request.json['metric']
    )
    # step 1: get dataset metafeatures
    t0 = time.time()
    metafeatures = get_dataset_metafeatures(recommendation_request)
    t1 = time.time()

    # step 2: get features concatenation of kpis with reference datasets
    metafeatures_df = build_metafeatures_df(
        metafeatures, recommendation_request.task)
    t2 = time.time()

    # step 3: run model on metafeatures_df, get top pipeline
    top_pipeline_id, top_pipeline_serialized = get_top_pipeline(
        metafeatures_df, recommendation_request.task, metric=recommendation_request.metric)
    t3 = time.time()

    print('get_dataset_metafeatures: {}s\nbuild_metafeatures_df: {}s\nget_top_pipeline: {}s'.format(
        t1-t0, t2-t1, t3-t2))

    return jsonify(
        pipeline_id=top_pipeline_id,
        pipeline=top_pipeline_serialized,
        # top_n_pipelines=top_n_pipelines
    )


if __name__ == '__main__':
    app.debug = True
    app.run(port=5000)
