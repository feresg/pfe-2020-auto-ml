from constants import Model, Preprocessor


class PipelinePreprocessor:
    def __init__(self, pipeline):
        '''
        Parses a pipeline object

        :pipeline: python dictionnary reporesentation of the pipeline
        '''
        self.pipeline = pipeline
        self.model = self._get_model()
        self.preprocessor = self._get_preprocessor()
        self.model_hyperparams = self._get_model_hyperparams()
        self.preprocessor_hyperparams = self._get_preprocessor_hyperparams()

    def _get_model(self):
        return Model(self.pipeline['model'])

    def _get_preprocessor(self):
        return Preprocessor(self.pipeline['pre-processor'])

    def _get_model_hyperparams(self):
        pass

    def _get_preprocessor_hyperparams(self):
        pass

    def generate_pipeline(self):
        '''
        generates a scikit-learn pipeline from the pipeline data model
        '''
        pass
