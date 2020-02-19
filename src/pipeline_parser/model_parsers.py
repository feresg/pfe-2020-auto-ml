def parse_libsvm_svc_hyperparams(pipeline):
    libsvm_svc_hyperparams = dict(
        kernel=pipeline['libsvm_svc_kernel'],
        C=pipeline['libsvm_svc_C'],
        max_iter=pipeline['libsvm_svc_max_iter'],
        tol=pipeline['libsvm_svc_tol'],
        shrinking=pipeline['libsvm_svc_shrinking'],
        gamma=pipeline['libsvm_svc_gamma'],
    )
    if 'libsvm_svc_coef0' in pipeline:
        libsvm_svc_hyperparams['coef0'] = pipeline['libsvm_svc_coef0']
    if 'libsvm_svc_degree' in pipeline:
        libsvm_svc_hyperparams['degree'] = pipeline['libsvm_svc_degree']
    return libsvm_svc_hyperparams


def parse_decision_tree_hyperparams(pipeline):
    return dict(
        splitter=pipeline['decision_tree_splitter'],
        min_samples_leaf=pipeline['decision_tree_min_samples_leaf'],
        max_features=pipeline['decision_tree_max_features'],
        min_weight_fraction_leaf=pipeline['decision_tree_min_weight_fraction_leaf'],
        criterion=pipeline['decision_tree_criterion'],
        min_samples_split=pipeline['decision_tree_min_samples_split'],
        max_leaf_nodes=None if pipeline['decision_tree_max_leaf_nodes'] < 2 else pipeline['decision_tree_max_leaf_nodes'],
        max_depth=pipeline['decision_tree_max_depth']
    )


def parse_random_forest_hyperparams(pipeline):
    # TODO: ignoring max_features hyperparam since it's causing a ValueError
    return dict(
        bootstrap=pipeline['random_forest_bootstrap'],
        min_samples_leaf=pipeline['random_forest_min_samples_leaf'],
        n_estimators=pipeline['random_forest_n_estimators'],
        # max_features=pipeline['random_forest_max_features'],
        min_weight_fraction_leaf=pipeline['random_forest_min_weight_fraction_leaf'],
        criterion=pipeline['random_forest_criterion'],
        min_samples_split=pipeline['random_forest_min_samples_split'],
        max_leaf_nodes=None if pipeline['random_forest_max_leaf_nodes'] < 2 else pipeline['random_forest_max_leaf_nodes'],
        max_depth=None if pipeline['random_forest_max_depth'] == -
        1 else pipeline['random_forest_max_depth']
    )


def parse_extra_trees_hyperparams(pipeline):
    # TODO: ignoring max_features hyperparam since it's causing a ValueError
    return dict(
        bootstrap=pipeline['extra_trees_bootstrap'],
        min_samples_leaf=pipeline['extra_trees_min_samples_leaf'],
        n_estimators=pipeline['extra_trees_n_estimators'],
        # max_features=pipeline['extra_trees_max_features'],
        min_weight_fraction_leaf=pipeline['extra_trees_min_weight_fraction_leaf'],
        criterion=pipeline['extra_trees_criterion'],
        min_samples_split=pipeline['extra_trees_min_samples_split'],
        max_depth=None if pipeline['extra_trees_max_depth'] == -
        1 else pipeline['extra_trees_max_depth']
    )


def parse_gradient_boosting_hyperparams(pipeline):
    # TODO: ignoring max_features hyperparam since it's causing a ValueError
    return dict(
        loss=pipeline['gradient_boosting_loss'],
        learning_rate=pipeline['gradient_boosting_learning_rate'],
        min_samples_leaf=pipeline['gradient_boosting_min_samples_leaf'],
        n_estimators=pipeline['gradient_boosting_n_estimators'],
        subsample=pipeline['gradient_boosting_subsample'],
        min_weight_fraction_leaf=pipeline['gradient_boosting_min_weight_fraction_leaf'],
        # max_features=pipeline['gradient_boosting_max_features'],
        min_samples_split=pipeline['gradient_boosting_min_samples_split'],
        max_depth=pipeline['gradient_boosting_max_depth'],
        max_leaf_nodes=None if pipeline['gradient_boosting_max_leaf_nodes'] < 2 else pipeline['gradient_boosting_max_leaf_nodes'],
    )


def parse_xgradient_boosting_hyperparams(pipeline):
    return dict(
        reg_alpha=pipeline['xgradient_boosting_reg_alpha'],
        colsample_bytree=pipeline['xgradient_boosting_colsample_bytree'],
        colsample_bylevel=pipeline['xgradient_boosting_colsample_bylevel'],
        scale_pos_weight=pipeline['xgradient_boosting_scale_pos_weight'],
        learning_rate=pipeline['xgradient_boosting_learning_rate'],
        max_delta_step=pipeline['xgradient_boosting_max_delta_step'],
        base_score=pipeline['xgradient_boosting_base_score'],
        n_estimators=pipeline['xgradient_boosting_n_estimators'],
        subsample=pipeline['xgradient_boosting_subsample'],
        reg_lambda=pipeline['xgradient_boosting_reg_lambda'],
        min_child_weight=pipeline['xgradient_boosting_min_child_weight'],
        max_depth=pipeline['xgradient_boosting_max_depth'],
        gamma=pipeline['xgradient_boosting_gamma']
    )


def parse_k_nearest_neighbors_hyperparams(pipeline):
    return dict(
        p=pipeline['k_nearest_neighbors_p'],
        weights=pipeline['k_nearest_neighbors_weights'],
        n_neighbors=pipeline['k_nearest_neighbors_n_neighbors']
    )


def parse_bernoulli_nb_hyperparams(pipeline):
    return dict(
        alpha=pipeline['bernoulli_nb_alpha'],
        fit_prior=pipeline['bernoulli_nb_fit_prior']
    )


def parse_multinomial_nb_hyperparams(pipeline):
    return dict(
        alpha=pipeline['multinomial_nb_alpha'],
        fit_prior=pipeline['multinomial_nb_fit_prior']
    )


def parse_gaussian_nb_hyperparams(pipeline):
    return dict()


def parse_lda_hyperparams(pipeline):
    # TODO: raises FutureWarning: In version 0.23, setting n_components > min(n_features, n_classes - 1) will raise a ValueError.
    #       raises ChangedBehaviorWarning: n_components cannot be larger than min(n_features, n_classes - 1)
    #       raises UserWarning: Variables are collinear
    lda_hyperparams = dict(
        n_components=pipeline['lda_n_components'],
        tol=pipeline['lda_tol']
    )
    if pipeline['lda_shrinkage'] != -1:
        lda_hyperparams['solver'] = 'lsqr'
    if pipeline['lda_shrinkage'] == 'auto':
        lda_hyperparams['shrinkage'] = 'auto'
    elif pipeline['lda_shrinkage'] == 'manual':
        lda_hyperparams['shrinkage'] = pipeline['lda_shrinkage_factor']
    return lda_hyperparams


def parse_qda_hyperparams(pipeline):
    return dict(
        reg_param=pipeline['qda_reg_param']
    )
