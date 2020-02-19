def parse_pca_hyperparams(pipeline):
    return dict(
        n_components=pipeline['pca_keep_variance'],
        whiten=pipeline['pca_whiten'],
        svd_solver='full'
    )


def parse_polynomial_hyperparams(pipeline):
    return dict(
        include_bias=pipeline['polynomial_include_bias'],
        interaction_only=pipeline['polynomial_interaction_only'],
        degree=pipeline['polynomial_degree']
    )
