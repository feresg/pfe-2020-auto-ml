from enum import Enum


class Model(Enum):
    LIBSVM_SVC = 'libsvm_svc'
    DECISION_TREE = 'decision_tree'
    RANDOM_FOREST = 'random_forest'
    EXTRA_TREES = 'extra_trees'
    GRADIENT_BOOSTING = 'gradient_boosting'
    XGRADIENT_BOOSTING = 'xgradient_boosting'
    K_NEAREST_NEIGHBORS = 'k_nearest_neighbors'
    BERNOULLI_NB = 'bernoulli_nb'
    MULTINOMIAL_NB = 'multinomial_nb'
    GAUSSIAN_NB = 'gaussian_nb'
    LDA = 'lda'
    QDA = 'qda'


class Preprocessor(Enum):
    PCA = 'pca'
    POLYNOMIAL = 'polynomial'
