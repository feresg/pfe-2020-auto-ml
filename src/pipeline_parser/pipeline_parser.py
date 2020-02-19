from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from xgboost import XGBClassifier

import model_parsers
import preprocessor_parsers


class PipelineParser:
    def __init__(self, pipeline):
        '''
        Parses a pipeline object

        :pipeline: python dictionnary reporesentation of the pipeline
        '''
        self.pipeline = pipeline
        self.model_name = pipeline['model']
        self.preprocessor_name = pipeline['pre-processor']
        self.model = self._get_model()
        self.preprocessor = self._get_preprocessor()
        self.model_hyperparams = self._get_model_hyperparams()
        self.preprocessor_hyperparams = self._get_preprocessor_hyperparams()

    def _get_model(self):
        # TODO: for now, we only consider classifiers, add regressors later
        options = dict(
            libsvm_svc=SVC,
            decision_tree=DecisionTreeClassifier,
            random_forest=RandomForestClassifier,
            extra_trees=ExtraTreesClassifier,
            gradient_boosting=GradientBoostingClassifier,
            xgradient_boosting=XGBClassifier,
            k_nearest_neighbors=KNeighborsClassifier,
            bernoulli_nb=BernoulliNB,
            multinomial_nb=MultinomialNB,
            gaussian_nb=GaussianNB,
            lda=LinearDiscriminantAnalysis,
            qda=QuadraticDiscriminantAnalysis
        )
        return options[self.model_name]

    def _get_preprocessor(self):
        options = dict(
            pca=PCA,
            polynomial=PolynomialFeatures
        )
        return options[self.preprocessor_name]

    def _get_model_hyperparams(self):
        options = dict(
            libsvm_svc=model_parsers.parse_libsvm_svc_hyperparams,
            decision_tree=model_parsers.parse_decision_tree_hyperparams,
            random_forest=model_parsers.parse_random_forest_hyperparams,
            extra_trees=model_parsers.parse_extra_trees_hyperparams,
            gradient_boosting=model_parsers.parse_gradient_boosting_hyperparams,
            xgradient_boosting=model_parsers.parse_xgradient_boosting_hyperparams,
            k_nearest_neighbors=model_parsers.parse_k_nearest_neighbors_hyperparams,
            bernoulli_nb=model_parsers.parse_bernoulli_nb_hyperparams,
            multinomial_nb=model_parsers.parse_multinomial_nb_hyperparams,
            gaussian_nb=model_parsers.parse_gaussian_nb_hyperparams,
            lda=model_parsers.parse_lda_hyperparams,
            qda=model_parsers.parse_qda_hyperparams
        )
        return options[self.model_name](self.pipeline)

    def _get_preprocessor_hyperparams(self):
        options = dict(
            pca=preprocessor_parsers.parse_pca_hyperparams,
            polynomial=preprocessor_parsers.parse_polynomial_hyperparams
        )
        return options[self.preprocessor_name](self.pipeline)

    def generate_pipeline(self):
        '''
        generates a scikit-learn pipeline from the pipeline data model
        '''
        return make_pipeline(
            self.preprocessor(**self.preprocessor_hyperparams),
            self.model(**self.model_hyperparams)
        )
