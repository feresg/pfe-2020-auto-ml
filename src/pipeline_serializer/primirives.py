class Primitive:

    def __init__(self, name):
        self.name = name


class Estimator(Primitive):
    def __init__(self, name, sklearn_estimator):
        super().__init__(name)
        self.parse_estimator(sklearn_estimator)

    def parse_estimator(self, sklearn_estimator):
        self.params = sklearn_estimator.get_params()


class Transformer(Primitive):
    def __init__(self, name, sklearn_transformer):
        super().__init__(name)
        self.parse_transformer(sklearn_transformer)

    def parse_transformer(self, sklearn_transformer):
        self.params = sklearn_transformer.get_params()
        self.params.pop('dtype', None)


class Pipeline(Primitive):

    def __init__(self, name, sklearn_pipeline):
        super().__init__(name)
        self.parse_pipeline(sklearn_pipeline)

    def parse_pipeline(self, sklearn_pipeline):
        self.steps = []
        for step in sklearn_pipeline.steps:
            step_name = step[0]
            step_block = step[1]

            if step_name in transformers:
                self.steps.append(Transformer(step_name, step_block))
            elif step_name in estimators:
                self.steps.append(Estimator(step_name, step_block))
            elif 'pipeline' in step_name:
                self.steps.append(Pipeline(step_name, step_block))
            elif 'featureunion' in step_name:
                self.steps.append(FeatureUnion(step_name, step_block))
            elif 'columntransformer' in step_name:
                self.steps.append(ColumnTransformer(step_name, step_block))


class FeatureUnion(Primitive):

    def __init__(self, name, sklearn_featureunion):
        super().__init__(name)
        self.parse_featureunion(sklearn_featureunion)

    def parse_featureunion(self, sklearn_featureunion):
        self.transformers = []
        for step in sklearn_featureunion.transformer_list:
            step_name = step[0]
            step_block = step[1]

            if step_name in transformers:
                self.transformers.append(Transformer(step_name, step_block))
            elif step_name in estimators:
                self.transformers.append(Estimator(step_name, step_block))
            elif 'pipeline' in step_name:
                self.transformers.append(Pipeline(step_name, step_block))
            elif 'featureunion' in step_name:
                self.transformers.append(FeatureUnion(step_name, step_block))
            elif 'columntransformer' in step_name:
                self.transformers.append(
                    ColumnTransformer(step_name, step_block))


class ColumnTransformer(Primitive):

    def __init__(self, name, sklearn_columntransformer):
        super().__init__(name)
        self.parse_columntransformer(sklearn_columntransformer)

    def parse_columntransformer(self, sklearn_columntransformer):
        self.transformers = []
        self.remainder = sklearn_columntransformer.remainder

        for transformer in sklearn_columntransformer.transformers:
            transformer_name = transformer[0]
            transformer_block = transformer[1]
            transformer_columnselector = transformer[2].dtype_include

            if transformer_name in transformers:
                self.transformers.append(
                    ColumnTransformerItem(
                        Transformer(transformer_name, transformer_block),
                        transformer_columnselector
                    )
                )
            elif 'pipeline' in transformer_name:
                self.transformers.append(
                    ColumnTransformerItem(
                        Pipeline(transformer_name, transformer_block),
                        transformer_columnselector
                    )
                )
            elif 'featureunion' in transformer_name:
                self.transformers.append(
                    ColumnTransformerItem(
                        FeatureUnion(transformer_name, transformer_block),
                        transformer_columnselector
                    )
                )


class ColumnTransformerItem:

    def __init__(self, transformer, column_selector):
        self.transformer = transformer
        self.column_selector = column_selector
