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

    'logisticregression': {}

}

regression_config = {
    'xgbregressor': {
        'xgbregressor__gamma': hp.loguniform('xgbregressor__gamma', 1e-8, 1.0),
        'xgbregressor__learning_rate': hp.loguniform('xgbregressor__learning_rate', 1e-4, 1e-1),
        'xgbregressor__booster': hp.choice('xgbregressor__booster', ['gbtree', 'dart']),
        'xgbregressor__n_estimators': hp.choice('xgbregressor__n_estimators', [*range(64, 513, 24)]),
        'xgbregressor__max_depth': hp.choice('xgbregressor__max_depth', [*range(3, 22, 2)]),
        'xgbregressor__max_features': hp.choice('xgbregressor__max_features', np.arange(0.5, 1.01, 0.05).tolist()),
        'xgbregressor__reg_alpha': hp.choice('xgbregressor__reg_alpha', [0, 0.01, 0.1, 0.5, 1]),
        'xgbregressor__reg_lambda': hp.choice('xgbregressor__reg_lambda', [0.001, 0.01, 0.1, 1, 10, 100]),
        'xgbregressor__colsample_bylevel': hp.uniform('xgbregressor__colsample_bylevel', 0.1, 1.0),
        'xgbregressor__colsample_bytree': hp.uniform('xgbregressor__colsample_bytree', 0.1, 1.0),
        'xgbregressor__subsample': hp.uniform('xgbregressor__subsample', 0.1, 1.0)
    },

    'lgbmregressor': {
        'lgbmregressor__boosting_type': hp.choice('lgbmregressor__boosting_type', ['gbdt', 'dart']),
        'lgbmregressor__n_estimators': hp.choice('lgbmregressor__n_estimators', [*range(64, 513, 24)]),
        'lgbmregressor__max_depth': hp.choice('lgbmregressor__max_depth', [-1, *range(3, 22, 2)]),
        'lgbmregressor__num_leaves': hp.choice('lgbmregressor__num_leaves', [*range(24, 129, 8)]),
        'lgbmregressor__reg_alpha': hp.choice('lgbmregressor__reg_alpha', [0, 0.01, 0.1, 0.5, 1]),
        'lgbmregressor__reg_lambda': hp.choice('lgbmregressor__reg_lambda', [0.001, 0.01, 0.1, 1, 10, 100])
    },


    'ardregression': {
        'ardregression__alpha_1': hp.loguniform('ardregression__alpha_1', 1e-10, 0.001),
        'ardregression__alpha_2': hp.loguniform('ardregression__alpha_2', 1e-10, 0.001),
        'ardregression__lambda_2': hp.loguniform('ardregression__lambda_2', 1e-10, 0.001),
        'ardregression__lambda_2': hp.loguniform('ardregression__lambda_2', 1e-10, 0.001),
        'ardregression__threshold_lambda': hp.loguniform('ardregression__threshold_lambda', 1000.0, 100000.0),
        'ardregression__tol': hp.loguniform('ardregression__tol', 1e-5, 0.1),
    },

    'kneighborsregressor': {
        'kneighborsregressor__weights': hp.choice('kneighborsregressor__weights', ['uniform', 'distance']),
        'kneighborsregressor__p': hp.choice('kneighborsregressor__p', [1, 2]),
        'kneighborsregressor__n_neighbors': hp.choice('kneighborsregressor__n_neighbors', [*range(2, 50)])
    },

    'gradientboostingregressor': {
        'gradientboostingregressor__learning_rate': hp.loguniform('gradientboostingregressor__learning_rate', 0.01, 1.0),
        'gradientboostingregressor__loss': hp.choice('gradientboostingregressor__loss', ['ls', 'lad', 'huber', 'quantile']),
        'gradientboostingregressor__max_features': hp.uniform('gradientboostingregressor__max_features', 0.1, 1.0),
        'gradientboostingregressor__min_samples_leaf': hp.uniform('gradientboostingregressor__main_samples_leaf', 0.0, 0.5),
        'gradientboostingregressor__min_samples_split': hp.uniform('gradientboostingregressor__main_samples_split', 0.0, 1.0),
        'gradientboostingregressor__n_estimators': hp.choice('gradientboostingregressor__n_estimators', [*range(64, 513, 24)]),
        'gradientboostingregressor__subsample': hp.uniform('gradientboostingregressor__subsample', 0.1, 1.0)
    },

    'extratreesregressor': {
        'extratreesregressor__max_features': hp.choice('extratreesregressor__max_features', ['sqrt', 'log2']),
        'extratreesregressor__bootstrap': hp.choice('extratreesregressor__bootstrap', [True, False]),
        'extratreesregressor__n_estimators': hp.choice('extratreesregressor__n_estimators', [*range(64, 513, 24)]),
        'extratreesregressor__max_depth': hp.choice('extratreesregressor__max_depth', [None, *range(3, 22, 2)]),
        'extratreesregressor__min_samples_leaf': hp.uniform('extratreesregressor__main_samples_leaf', 0.0, 0.5),
        'extratreesregressor__min_samples_split': hp.uniform('extratreesregressor__min_samples_split', 0.0, 1.0),
        'extratreesregressor__max_features': hp.choice('extratreesregressor__max_features', ['sqrt', 'log2'])
    },

    'randomforestregressor': {
        'randomforestregressor__criterion': hp.choice('randomforestregressor__criterion', ['mse', 'friedman_mse', 'mae']),
        'randomforestregressor__max_features': hp.choice('randomforestregressor__max_features', ['sqrt', 'log2']),
        'randomforestregressor__bootstrap': hp.choice('randomforestregressor__bootstrap', [True, False]),
        'randomforestregressor__n_estimators': hp.choice('randomforestregressor__n_estimators', [*range(64, 513, 24)]),
        'randomforestregressor__max_depth': hp.choice('randomforestregressor__max_depth', [None, *range(3, 22, 2)]),
        'randomforestregressor__min_samples_split': hp.uniform('randomforestregressor__min_samples_split', 0.0, 1.0),
        'randomforestregressor__min_samples_leaf': hp.uniform('randomforestregressor__min_samples_split', 0.0, 0.5)
    },

    'decisiontreeregressor': {
        'decisiontreeregressor__criterion': hp.choice('decisiontreeregressor__criterion', ['mse', 'friedman_mse', 'mae']),
        'decisiontreeregressor__max_features': hp.choice('decisiontreeregressor__max_features', ['sqrt', 'log2']),
        'decisiontreeregressor__max_depth': hp.choice('decisiontreeregressor__max_depth', [None, *range(3, 22, 2)]),
        'decisiontreeregressor__min_samples_split': hp.uniform('decisiontreeregressor__min_samples_split', 0.0, 1.0),
        'decisiontreeregressor__min_samples_leaf': hp.uniform('decisiontreeregressor__min_samples_split', 0.0, 0.5)
    },

    'gaussianprecessregressor': {
        'gaussianprocessregressor__alpha': hp.loguniform('gaussianprocessregressor__alpha', 1e-14, 1.0),
    },

    'ridge': {
        'ridge__alpha': hp.loguniform('gaussianprocessregressor__alpha', 1e-5, 10.0),
        'ridge__tol': hp.loguniform('gaussianprocessregressor__alpha', 1e-5, 0.1)
    },

    'svr': {
        'svr__tol': hp.loguniform('svr__tol', 1e-05, 0.1),
        'svr__epsilon': hp.loguniform('svr__C', 0.001, 1.0),
        'svr__C': hp.loguniform('svr__C', 0.03125, 32768.0),
        'svr__kernel': hp.choice('svr__kernel', ['linear', 'rbf', 'poly', 'sigmoid']),
        'svr__shrinking': hp.choice('svr__shrinking', [True, False]),
    },

    'linearsvr': {
        'linearsvr__tol': hp.loguniform('linearsvr__tol', 0.00001, 0.1),
        'linearsvr__C': hp.loguniform('linearsvr__C', 0.03125, 32768.0),
        'linearsvr__epsilon': hp.loguniform('linearsvr__C', 0.01, 1.0),
    },

    'sgdregressor': {
        'sgdregressor__alpha': hp.loguniform('sgdregressor__alpha', 1e-7, 0.1),
        'sgdregressor__tol': hp.loguniform('sgdregressor__tol', 1e-5, 0.1),
        # 'sgdregressor__eta0': hp.loguniform('sgdregressor__eta0', 1e-7, 0.1),
        # 'sgdregressor__epsilon': hp.loguniform('sgdregressor__epsilon', 1e-5, 0.1),
        # 'sgdregressor__power_t': hp.uniform('sgdregressor__power_t', 1e-5, 1.0),
        'sgdregressor__average': hp.choice('sgdregressor__average', [True, False]),
        'sgdregressor__learning_rate': hp.choice('sgdregressor__learning_rate', ['optimal', 'invscaling', 'constant']),
        'sgdregressor__loss': hp.choice('sgdregressor__loss', ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
        'sgdregressor__penalty': hp.choice('sgdregressor__penalty', ['l1', 'l2', 'elasticnet'])
    },



}
