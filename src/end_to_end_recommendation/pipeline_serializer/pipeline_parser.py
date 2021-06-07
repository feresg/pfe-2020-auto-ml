import simplejson as json
from .utils import todict
from .primirives import Pipeline


def parse_pipeline_to_json(sklearn_pipeline):
    pipeline = Pipeline('', sklearn_pipeline)
    pipeline_dict = todict(pipeline)
    pipeline_json = json.dumps(pipeline_dict, indent=4, ignore_nan=True)
    return pipeline_json
