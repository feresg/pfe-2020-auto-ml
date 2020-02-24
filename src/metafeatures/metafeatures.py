import pandas as pd

from statistical_metafeatures import StatisticalMetafeaturesComputer
from info_theoretical_metafeatures import InfoTheoreticalMetafeatures
from landmarking_metafeatures import LandmarkingMetafeatures
from descriptive_metafeatures import DescriptiveMetafeatures
from constants import Task, Heuristic
from utils import timeit

mf_heuristics = {
    Heuristic.STATISTICAL: StatisticalMetafeaturesComputer,
    Heuristic.INFO_THEORETICAL: InfoTheoreticalMetafeatures,
    Heuristic.LANDMARKING: LandmarkingMetafeatures,
    Heuristic.DESCRIPTIVE: DescriptiveMetafeatures
}


class BaseMetafeaturesComputer:
    def __init__(self):
        pass

    @staticmethod
    def compute(X, y, task):
        pass


class Metafeatures:

    def __init__(self, df, target, heuristics=None, task=None):
        # TODO: assert that df is of type Pandas dataframe, that target is a column in df
        self.y = df[target]
        self.X = df.drop(labels=target, axis=1)
        # TODO: check that heuristics and task values set by user are valid
        self.heuristics = heuristics if heuristics else [
            Heuristic.DESCRIPTIVE, Heuristic.STATISTICAL, Heuristic.INFO_THEORETICAL, Heuristic.LANDMARKING]
        self.task = Task(task) if task is not None else self._get_task()

    @timeit
    def compute(self):
        # TODO: compute all requested metafeatures and return them as dictionary
        metafeatures = {}
        for heuristic in self.heuristics:
            metafeatures.update(mf_heuristics[heuristic].compute(
                self.X, self.y, self.task))
        return metafeatures

    def _get_task(self):
        return Task.CLASSIFICATION if pd.api.types.is_categorical_dtype(self.y) else Task.REGRESSION
