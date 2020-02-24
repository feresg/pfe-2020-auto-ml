from metafeatures import BaseMetafeaturesComputer
from constants import Task


class DescriptiveMetafeatures(BaseMetafeaturesComputer):
    def __init__(self):
        super().__init__(self)

    @staticmethod
    def compute(X, y, task):
        metafeatures = {}
        metafeatures['nb_instances'] = X.shape[0]
        metafeatures['nb_features'] = X.shape[1]
        if task == Task.CLASSIFICATION:
            # TODO: add majority, minority class size, nb classes...
            pass
        return metafeatures
