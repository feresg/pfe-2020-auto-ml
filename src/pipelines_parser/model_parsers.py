def parse_libsvm_svc_hyperparams(pipeline):
    libsvm_svc_params = dict(
        kernel=pipeline['libsvm_svc_kernel'],
        C=pipeline['libsvm_svc_C'],
        max_iter=pipeline['libsvm_svc_max_iter'],
        coef0=pipeline['libsvm_svc_coef0'],
        tol=pipeline['libsvm_svc_tol'],
        shrinking=pipeline['libsvm_svc_shrinking'],
        gamma=pipeline['libsvm_svc_gamma'],
    )
    if 'libsvm_svc_coef0' in pipeline:
        libsvm_svc_params['coef0'] = pipeline['libsvm_svc_coef0']
    if 'libsvm_svc_degree' in pipeline:
        libsvm_svc_params['degree'] = pipeline['degree']
    return libsvm_svc_params


def parse_decision_tree_hyperparams(pipeline):
    return dict(
        splitter=pipeline['decision_tree_splitter'],
        max_leaf_nodes=pipeline['decision_tree_max_leaf_nodes'],
        min_samples_leaf=pipeline['decision_tree_min_samples_leaf'],
        max_features=pipeline['decision_tree_max_features'],
        criterion=pipeline['decision_tree_criterion'],
        min_samples_split=pipeline['decision_tree_min_samples_split'],
        max_depth=pipeline['decision_tree_max_depth']
    )


def parse_random_forest_hyperparams(pipeline):
    return dict(
        bootstrap=pipeline['random_forest_bootstrap'],
        max_leaf_nodes=pipeline['min_samples_leaf'],
        min_samples_leaf=pipeline['random_forest_min_samples_leaf'],
        n_estimators=pipeline['random_forest_n_estimators'],
        max_features=pipeline['random_forest_max_features'],
        min_weight_fraction_leaf=pipeline['random_forest_min_weight_fraction_leaf'],
        criterion=pipeline['random_forest_criterion'],
        min_samples_split=pipeline['random_forest_min_samples_split'],
        max_depth=pipeline['random_forest_max_depth']
    )


def parse_extra_trees_hyperparams(pipeline):
    pass


def parse_gradient_boosting_hyperparams(pipeline):
    pass


def parse_xgradient_boosting_hyperparams(pipeline):
    pass


def parse_k_nearest_neighbors_hyperparams(pipeline):
    pass


def parse_bernoulli_nb_hyperparams(pipeline):
    pass


def parse_multinominal_nb_hyperparams(pipeline):
    pass


def parse_gaussian_nb_hyperparams(pipeline):
    return dict()


def parse_lda_hyperparams(pipeline):
    pass


def parse_qda_hyperparams(pipeline):
    return dict(
        reg_param=pipeline['qda_reg_param']
    )
