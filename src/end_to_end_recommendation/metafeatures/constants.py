from enum import Enum
import numpy as np


class Task(Enum):
    # TODO: add forecasting
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'


class Heuristic(Enum):
    STATISTICAL = 'statistical'
    INFO_THEORETICAL = 'info_theoretical'
    DESCRIPTIVE = 'descriptive'
    LANDMARKING = 'landmarking'
    CORRELATION = 'correlation'


empty_dataframe_stats = [np.nan, np.nan,
                         np.nan, np.nan, np.nan, np.nan, np.nan]

empty_dataframe_stats_no_q = [np.nan, np.nan, np.nan, np.nan, np.nan]
