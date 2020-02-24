from enum import Enum


class Task(Enum):
    # TODO: add forecasting
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'


class Heuristic(Enum):
    STATISTICAL = 'statistical'
    INFO_THEORETICAL = 'info_theoretical'
    DESCRIPTIVE = 'descriptive'
    LANDMARKING = 'landmarking'
