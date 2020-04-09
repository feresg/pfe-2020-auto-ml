import pandas as pd

from statistical_metafeatures import StatisticalMetafeaturesComputer
from info_theoretical_metafeatures import InfoTheoreticalMetafeatures
from landmarking_metafeatures import LandmarkingMetafeatures
from descriptive_metafeatures import DescriptiveMetafeatures
from correlation_metafeatures import CorrelationMetafeatures
from constants import Task, Heuristic
from utils import timeit


mf_heuristics = {
    Heuristic.STATISTICAL: StatisticalMetafeaturesComputer,
    Heuristic.INFO_THEORETICAL: InfoTheoreticalMetafeatures,
    Heuristic.LANDMARKING: LandmarkingMetafeatures,
    Heuristic.DESCRIPTIVE: DescriptiveMetafeatures,
    Heuristic.CORRELATION: CorrelationMetafeatures
}


class Metafeatures:

    def __init__(self, X, y, heuristics=None, task=None):
        self.X = X
        self.y = y

        # TODO: check that heuristics and task values set by user are valid
        self.heuristics = heuristics if heuristics else [
            Heuristic.DESCRIPTIVE,
            Heuristic.STATISTICAL,
            # TODO: check why info theoretical metafeatures return too many null values
            Heuristic.INFO_THEORETICAL,
            Heuristic.LANDMARKING,
            Heuristic.CORRELATION
        ]
        self.task = Task(task) if task else self._get_task()

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
