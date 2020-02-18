def parse_pca_hyperparams(pipeline):
    return dict(
        keep_variance=pipeline['pca_keep_variance'],
        whiten=pipeline['pca_whiten']
    )


def parse_polynomial_hyperparams(pipeline):
    return dict(
        include_bias=pipeline['polynomial_include_bias'],
        interaction_only=pipeline['polynomial_interaction_only'],
        degree=pipeline['degree']
    )
