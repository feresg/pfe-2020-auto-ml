from enum import Enum


class Task(Enum):
    # TODO: add forecasting
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'


class Heuristic(Enum):
    PROBABILISTIC = 'probabilistic'
