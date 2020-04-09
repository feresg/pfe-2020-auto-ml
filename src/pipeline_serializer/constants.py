TRANSFORMERS = {'pca', 'truncatedsvd', 'variancethreshold', 'simpleimputer', 'ordinalencoder', 'polynomialfeatures',
                'standardscaler', 'minmaxscaler', 'onehotencoder', 'normalizer', 'robustscaler', 'rbfsampler',
                'featureagglomeration'}

CLASSIFIERS = {'baggingclassifier', 'decisiontreeclassifier', 'extratreesclassifier', 'gaussiannb', 'kneighborsclassifier',
               'lgbmclassifier', 'linearsvc', 'logisticregression', 'quadraticdiscriminantanalysis', 'randomforestclassifier',
               'sgdclassifier', 'svc', 'xgbclassifier'}

REGRESSORS = {'lgbmregressor', 'kneighborsregressor', 'xgbregressor', 'randomforestregressor', 'ardregression',
              'gradientboostingregressor', 'decisiontreeregressor', 'gaussianprocessregressor', 'extratreesregressor',
              'linearsvr', 'svr', 'ridge', 'sgdregressor'}


ESTIMATORS = set().union(CLASSIFIERS, REGRESSORS)
