from hyperopt import hp
import numpy as np

classification_config = {
    'xgbclassifier': {
        'xgbclassifier__gamma': hp.loguniform('xgbclassifier__gamma', 1e-8, 1.0),
        # 'xgbclassifier__learning_rate': hp.loguniform('xgbclassifier__learning_rate', 1e-4, 1e-1),
        'xgbclassifier__booster': hp.choice('xgbclassifier__booster', ['gbtree', 'dart']),
        'xgbclassifier__criterion': hp.choice('xgbclassifier__criterion', ['gini', 'entropy']),
        'xgbclassifier__bootstrap': hp.choice('xgbclassifier__bootstrap', [True, False]),
        'xgbclassifier__n_estimators': hp.choice('xgbclassifier__n_estimators', [*range(64, 513, 24)]),
        'xgbclassifier__max_depth': hp.choice('xgbclassifier__max_depth', [*range(3, 22, 2)]),
        'xgbclassifier__max_features': hp.choice('xgbclassifier__max_features', np.arange(0.5, 1.01, 0.05).tolist()),
        'xgbclassifier__reg_alpha': hp.choice('xgbclassifier__reg_alpha', [0, 0.01, 0.1, 0.5, 1]),
        'xgbclassifier__reg_lambda': hp.choice('xgbclassifier__reg_lambda', [0.001, 0.01, 0.1, 1, 10, 100])
    },

    'lgbmclassifier': {
        'lgbmclassifier__boosting_type': hp.choice('lgbmclassifier__boosting_type', ['gbdt', 'dart']),
        'lgbmclassifier__n_estimators': hp.choice('lgbmclassifier__n_estimators', [*range(64, 513, 24)]),
        'lgbmclassifier__max_depth': hp.choice('lgbmclassifier__max_depth', [-1, *range(3, 22, 2)]),
        'lgbmclassifier__num_leaves': hp.choice('lgbmclassifier__num_leaves', [*range(24, 129, 8)]),
        'lgbmclassifier__reg_alpha': hp.choice('lgbmclassifier__reg_alpha', [0, 0.01, 0.1, 0.5, 1]),
        'lgbmclassifier__reg_lambda': hp.choice('lgbmclassifier__reg_lambda', [0.001, 0.01, 0.1, 1, 10, 100])
    },

    'decisiontreeclassifier': {
        'decisiontreeclassifier__criterion': hp.choice('decisiontreeclassifier__criterion', ['gini', 'entropy']),
        'decisiontreeclassifier__max_depth': hp.choice('decisiontreeclassifier__max_depth', [None, *range(3, 22, 2)]),
        'decisiontreeclassifier__max_features': hp.choice('decisiontreeclassifier__max_features', ['sqrt', 'log2']),
        'decisiontreeclassifier__min_samples_split': hp.uniform('decisiontreeclassifier__min_samples_split', 0.0, 1.0)
    },

    'randomforestclassifier': {
        'randomforestclassifier__criterion': hp.choice('randomforestclassifier__criterion', ['gini', 'entropy']),
        'randomforestclassifier__max_features': hp.choice('randomforestclassifier__max_features', ['sqrt', 'log2']),
        'randomforestclassifier__bootstrap': hp.choice('randomforestclassifier__bootstrap', [True, False]),
        'randomforestclassifier__n_estimators': hp.choice('randomforestclassifier__n_estimators', [*range(64, 513, 24)]),
        'randomforestclassifier__max_depth': hp.choice('randomforestclassifier__max_depth', [None, *range(3, 22, 2)]),
        'randomforestclassifier__min_samples_split': hp.uniform('randomforestclassifier__min_samples_split', 0.0, 1.0)
    },

    'extratreesclassifier': {
        'extratreesclassifier__criterion': hp.choice('extratreesclassifier__criterion', ['gini', 'entropy']),
        'extratreesclassifier__max_features': hp.choice('extratreesclassifier__max_features', ['sqrt', 'log2']),
        'extratreesclassifier__bootstrap': hp.choice('extratreesclassifier__bootstrap', [True, False]),
        'extratreesclassifier__n_estimators': hp.choice('extratreesclassifier__n_estimators', [*range(64, 513, 24)]),
        'extratreesclassifier__max_depth': hp.choice('extratreesclassifier__max_depth', [None, *range(3, 22, 2)]),
        'extratreesclassifier__min_samples_split': hp.uniform('extratreesclassifier__min_samples_split', 0.0, 1.0)
    },

    'gradientboostingclassifier': {},

    'baggingclassifier': {
        'baggingclassifier__bootstrap': hp.choice('baggingclassifier__bootstrap', [True, False]),
        'baggingclassifier__n_estimators': hp.choice('baggingclassifier__n_estimators', [*range(64, 513, 24)]),
        'baggingclassifier__max_features': hp.uniform('baggingclassifier__max_features', 0.1, 1.0)
    },

    'kneighborsclassifier': {
        'kneighborsclassifier__weights': hp.choice('kneighborsclassifier__weights', ['uniform', 'distance']),
        'kneighborsclassifier__p': hp.choice('kneighborsclassifier__p', [1, 2]),
        'kneighborsclassifier__n_neighbors': hp.choice('kneighborsclassifier__n_neighbors', [*range(2, 25)])
    },

    'gaussiannb': {
        'gaussiannb__var_smoothing': hp.loguniform('gaussiannb__var_smoothing', 1e-11, 1e-7)
    },

    'quadraticdiscriminantanalysis': {
        'quadraticdiscriminantanalysis__reg_param': hp.uniform('quadraticdiscriminantanalysis__reg_param', 0.0, 1.0),
        'quadraticdiscriminantanalysis__tol': hp.uniform('quadraticdiscriminantanalysis__tol', 1e-6, 1e-2)
    },

    'sgdclassifier': {
        'sgdclassifier__alpha': hp.loguniform('sgdclassifier__alpha', 1e-7, 0.1),
        'sgdclassifier__tol': hp.loguniform('sgdclassifier__tol', 1e-5, 0.1),
        'sgdclassifier__eta0': hp.loguniform('sgdclassifier__eta0', 1e-7, 0.1),
        # 'sgdclassifier__epsilon': hp.loguniform('sgdclassifier__epsilon', 1e-5, 0.1),
        # 'sgdclassifier__power_t': hp.uniform('sgdclassifier__power_t', 1e-5, 1.0),
        'sgdclassifier__average': hp.choice('sgdclassifier__average', [True, False]),
        'sgdclassifier__learning_rate': hp.choice('sgdclassifier__learning_rate', ['optimal', 'invscaling', 'constant']),
        'sgdclassifier__loss': hp.choice('sgdclassifier__loss', ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']),
        'sgdclassifier__penalty': hp.choice('sgdclassifier__penalty', ['l1', 'l2', 'elasticnet'])
    },

    'svc': {
        'svc__tol': hp.loguniform('svc__tol', 0.00001, 0.1),
        'svc__C': hp.loguniform('svc__C', 0.03125, 32768.0),
        'svc__kernel': hp.choice('svc__kernel', ['linear', 'rbf', 'poly', 'sigmoid']),
        'svc__shrinking': hp.choice('svc__shrinking', [True, False]),
    },

    'linearsvc': {
        'linearsvc__tol': hp.loguniform('linearsvc__tol', 0.00001, 0.1),
        'linearsvc__C': hp.loguniform('linearsvc__C', 0.03125, 32768.0),
    },

    'linearregression': {}

}
