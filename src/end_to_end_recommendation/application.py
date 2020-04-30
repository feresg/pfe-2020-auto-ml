from utils import get_dataset_metafeatures, build_metafeatures_df, get_top_pipeline
from enum import Enum
from flask import Flask, request, jsonify

app = Flask(__name__)


class RecommendationRequest:

    def __init__(self, dataset_url, target_feature, task):
        self.dataset_url = dataset_url
        self.target_feature = target_feature
        self.task = task

# TODO: return http error status if task is not in [classification, regression], or invalid dataset url or if target_feature not in dataset
@app.route('/recommend', methods=['GET'])
def recommend_pipeline():
    # request data contains dataset path/url, target feature name and task type (classification/regression)
    recommendation_request = RecommendationRequest(
        dataset_url=request.json['dataset_url'],
        target_feature=request.json['target_feature'],
        task=request.json['task']
    )
    # step 1: get dataset metafeatures
    metafeatures = get_dataset_metafeatures(recommendation_request)

    # step 2: get features concatenation of kpis with reference datasets
    metafeatures_df = build_metafeatures_df(
        metafeatures, recommendation_request.task)

    # step 3: run model on metafeatures_df, get top pipeline
    top_pipeline_serialized = get_top_pipeline(
        metafeatures_df, recommendation_request.task)

    return jsonify(
        pipeline=top_pipeline_serialized
    )


if __name__ == '__main__':
    app.debug = True
    app.run(port=5001)
