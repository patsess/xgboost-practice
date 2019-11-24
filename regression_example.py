
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (RepeatedKFold, GridSearchCV,
    RandomizedSearchCV, ParameterGrid, ParameterSampler)
import xgboost as xgb
import lightgbm as lgb
from bayes_search_cv_class import BayesSearchCV_

# TODO: investigate HyperOpt package?


class XgboostExample(object):
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info('initialising a {} instance'
                         .format(self.__class__.__name__))

        # helpers for properties
        self._random_state = None
        self._default_grid_search_params = None
        self._default_randomised_search_params = None
        self._default_bayes_search_params = None

        # initialise attributes
        self._boston = None  # raw data from sklearn
        self._data = None  # pandas data frame
        self._data_dmatrix = None  # xgboost data structure
        self._X = None  # numpy array for features
        self._y = None  # numpy array for target variable
        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None

    @property
    def random_state(self):
        if self._random_state is None:
            self._random_state = 666

        return self._random_state

    @property
    def default_grid_search_params(self):
        if self._default_grid_search_params is None:
            self._default_grid_search_params = {
                'learning_rate': [0.1],
                'min_split_loss': [0.],  # min loss reduction req for split
                'colsample_bytree': [0.5],  # subsample ratio of column
                'subsample': [0.7],  # subsample ratio of the training samples
                'max_depth': [3, 5],
                'n_estimators': [100, 200],
                'reg_alpha': [0., 1., 10.],  # L1 reg term on weights
                'reg_lambda': [0., 1., 10.],  # L2 reg term on weights
            }
            # note: alias for 'min_split_loss' is 'gamma'

        return self._default_grid_search_params

    @property
    def default_randomised_search_params(self):
        if self._default_randomised_search_params is None:
            self._default_randomised_search_params = {
                'learning_rate': [0.05, 0.1, 0.2],
                'min_split_loss': [0., 1.],
                'colsample_bytree': [0.5, 0.6],
                'subsample': [0.7, 0.8],
                'max_depth': range(2, 9),
                'n_estimators': [100, 200],
                'reg_alpha': np.geomspace(0.001, 20., num=10).tolist(),
                'reg_lambda': np.geomspace(0.001, 20., num=10).tolist(),
            }

        return self._default_randomised_search_params

    @property
    def default_bayes_search_params(self):
        if self._default_bayes_search_params is None:
            self._default_bayes_search_params = {
                'learning_rate': (0.05, 0.35, 'log-uniform'),
                'gamma': (1e-9, 1., 'log-uniform'),  # default: 0.
                'min_child_weight': (1, 10),  # default: 1
                'max_delta_step': (0, 10),  # default: 0
                'max_depth': (2, 8),
                'colsample_bytree': (0.2, 1., 'uniform'),
                'colsample_bylevel': (0.2, 1., 'uniform'),
                'subsample': (0.2, 1., 'uniform'),
                'reg_alpha': (1e-9, 100., 'log-uniform'),
                'reg_lambda': (1e-9, 100., 'log-uniform'),
                # 'scale_pos_weight': (1e-6, 500, 'log-uniform'),  # default: 1
                'n_estimators': (20, 1000),
            }

        return self._default_bayes_search_params

    def load_data(self):
        """Sets attributes to store data"""
        self.logger.info('loading data...')

        # load data (via an sklearn data class)
        boston = load_boston()

        # examine structure of data
        self.logger.info('keys of data class: {}'.format(boston.keys()))
        self.logger.info('shape of data: {}'.format(boston.data.shape))
        self.logger.info('feature names: {}'.format(boston.feature_names))
        self.logger.info('description of data class:\n{}'.format(boston.DESCR))

        # convert data to a pandas data frame
        data = pd.DataFrame(boston.data)
        data.columns = boston.feature_names
        data['PRICE'] = boston.target  # target variable
        self.logger.info('\n{}'.format(data.head()))  # view data frame

        # also store data as xgboost's optimised data structure
        X, y = data.iloc[:, :-1], data.iloc[:, -1]
        data_dmatrix = xgb.DMatrix(data=X, label=y)

        # get train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state)

        # set attributes
        self._boston = boston
        self._data = data
        self._data_dmatrix = data_dmatrix
        self._X = X
        self._y = y
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test
        self.logger.info('...data loaded')

    def model_with_sklearn_api(self, cv_routine='grid_search', param_grid=None,
                               plot=True, use_lightgbm=False):
        self.logger.info('modelling with sklearn api...')

        # specify model using sklearn api
        if use_lightgbm:
            xg_reg = lgb.LGBMRegressor(objective='regression',
                                       random_state=self.random_state)
        else:
            xg_reg = xgb.XGBRegressor(objective='reg:linear',
                                      random_state=self.random_state)
        # note: at time of writing at least (2018-12), it seems that early
        # stopping (e.g. by adding early_stopping_rounds=50) is not supported
        # with sklearn's hyper-parameter optimisers like GridSearchCV

        # note: xgb.XGBRegressor and lgb.LGBMRegressor support many types of
        # regressions (i.e. many different loss functions), e.g. Poisson
        # regression, via the 'objective' argument

        if param_grid is None:
            param_grid = self._get_default_param_gird(cv_routine=cv_routine,
                                                      use_lightgbm=use_lightgbm)

        xg_reg = self._fit_model(xg_reg=xg_reg, cv_routine=cv_routine,
                                 param_grid=param_grid)

        if cv_routine in ['grid_search', 'randomised_search']:
            # use xgboot's api to further optimise n_estimators early stopping
            # note: no scaling here (but that is fine for tree weak learners)
            xg_reg = self._update_n_estimators(est=xg_reg,
                                               use_lightgbm=use_lightgbm)
        else:
            assert (cv_routine in ['bayes_search'])
            # note: Bayes searches should be capable of optimising
            # n_estimators well without early stopping

        # get predictions for test set
        preds = xg_reg.predict(self._X_test)

        # get root mean-square-error in test set
        rmse = np.sqrt(mean_squared_error(y_true=self._y_test, y_pred=preds))
        self.logger.info('RMSE in test set: {}'.format(rmse))

        if plot:
            # visualise model (using xgboost functionality)
            xgb.plot_tree(xg_reg.named_steps['model'], num_trees=0)
            plt.rcParams['figure.figsize'] = [50, 10]
            plt.show()

            xgb.plot_importance(xg_reg.named_steps['model'])
            plt.rcParams['figure.figsize'] = [5, 5]
            plt.show()
            # TODO: ensure names of features are used

        return xg_reg

    def _get_default_param_gird(self, cv_routine, use_lightgbm):
        if cv_routine == 'grid_search':
            param_grid = self.default_grid_search_params
        elif cv_routine == 'randomised_search':
            param_grid = self.default_randomised_search_params
        elif cv_routine == 'bayes_search':
            param_grid = self.default_bayes_search_params
        else:
            raise ValueError('unrecognised cv_routine {}'.format(cv_routine))

        if use_lightgbm:
            assert ('gamma' not in param_grid)  # should be 'min_split_loss'
            if 'min_split_loss' in param_grid:
                param_grid['min_split_gain'] = param_grid['min_split_loss']
                del param_grid['min_split_loss']
                # note: different name for same param

            if 'colsample_bylevel' in param_grid:
                del param_grid['colsample_bylevel']
                # note: param seems not to be present in LightGBM

            if 'max_delta_step' in param_grid:
                del param_grid['max_delta_step']
                # note: param seems not to be present in LightGBM

            # note: param 'min_child_samples' (perhaps among others) is
            # in LightGBM but an equivalent in xgboost

        return param_grid

    def _fit_model(self, xg_reg, cv_routine, param_grid):
        # if applicable, set random states
        xg_reg = self._set_estimator_random_state(est=xg_reg)

        # set up pipeline: scaling then regressing
        pipeline_list = [
            # ('scaler', StandardScaler()),  # unnecessary for trees
            ('model', xg_reg)
        ]

        one_cv_param = all([len(p) == 1 for p in param_grid.values()])
        if one_cv_param:
            # one candidate-set of params, so no cv required
            self.logger.info('only one set of hyper-parameters specified, '
                             'so no cross validation will be done')
            xg_reg.set_params(**{p: v[0] for p, v in param_grid.items()})
            xg_reg_cv = Pipeline(pipeline_list)
        else:
            # set up param_grid for cv
            param_grid = {'model__{}'.format(k): v for k, v in
                          param_grid.items()}

            # get cv instance
            xg_reg_cv = self._get_estimator_cv_instance(
                cv_routine=cv_routine, pipeline_list=pipeline_list,
                param_grid=param_grid)

        # fit pipeline
        self.logger.info('fitting (n_iter {}) pipeline with hyper-parameter '
                         'candidates {}'.format(xg_reg_cv.n_iter, param_grid))
        xg_reg_cv.fit(X=self._X_train, y=self._y_train)

        if one_cv_param:
            xg_reg = xg_reg_cv
        else:
            self.logger.info('pipeline fitted with hyper-parameters: {}'
                             .format(xg_reg_cv.best_params_))
            xg_reg = xg_reg_cv.best_estimator_

        return xg_reg

    def _set_estimator_random_state(self, est):
        if hasattr(est, 'random_state'):
            self.logger.info('setting random state of estimator (to {})'
                             .format(self.random_state))
            est.set_params(random_state=self.random_state)
            assert (est.random_state == self.random_state)

        if (hasattr(est, 'base_estimator') and
            hasattr(est.base_estimator, 'random_state')):
            # set random state in any base estimators
            self.logger.info('setting random state of base estimators (to {})'
                             .format(self.random_state))
            est.base_estimator.set_params(random_state=self.random_state)
            assert (est.base_estimator.random_state == self.random_state)

        return est

    def _get_estimator_cv_instance(self, cv_routine, pipeline_list,
                                   param_grid):
        kf = RepeatedKFold(n_splits=3, n_repeats=10,
                           random_state=self.random_state)

        if cv_routine == 'grid_search':
            est_cv = GridSearchCV(
                Pipeline(pipeline_list), param_grid,
                cv=kf, refit=True, n_jobs=1, verbose=0)

        elif cv_routine == 'randomised_search':
            n_iter = np.prod(
                [len(p) for p in self.default_grid_search_params.values()])
            est_cv = RandomizedSearchCV(
                Pipeline(pipeline_list), param_grid, n_iter=n_iter,
                cv=kf, refit=True, n_jobs=1, verbose=0,
                random_state=self.random_state)

        elif cv_routine == 'bayes_search':
            n_iter = np.prod(
                [len(p) for p in self.default_grid_search_params.values()])
            opt_kwargs = {'base_estimator': 'GP'}  # e.g. RF for random forest
            # note: default algorithm for surrogate is Gaussian Processes (GP)
            est_cv = BayesSearchCV_(
                Pipeline(pipeline_list), param_grid, n_iter=n_iter,
                cv=kf, refit=True, n_jobs=1, verbose=0,
                random_state=self.random_state, optimizer_kwargs=opt_kwargs)

        else:
            raise ValueError('unrecognised cv_routine {}'.format(cv_routine))

        return est_cv

    def _update_n_estimators(self, est, plot=False, use_lightgbm=False):
        # note: xgb.cv and lgb.cv (below) are for given hyper-parameters

        if use_lightgbm:
            self.logger.info('not update using early stopping done because '
                             'the does not seem to work with LightGBM')
            return est
            # TODO: it doesn't seem that early stopping works with LightGBM?

            lgb_cv_params = est.named_steps['model'].get_params().copy()
            if 'n_estimators' in lgb_cv_params:
                del lgb_cv_params['n_estimators']  # remove for early stopping

            lgb_cv_dataset = lgb.Dataset(data=self._X_train,
                                         label=self._y_train)
            cv_results = lgb.cv(
                train_set=lgb_cv_dataset, params=lgb_cv_params, nfold=3,
                num_boost_round=2000, early_stopping_rounds=50, metrics='rmse',
                seed=self.random_state, stratified=False)
            # note: stratified==False for regression
            result_name = 'rmse-mean'
            cv_results[result_name] = pd.Series(cv_results[result_name])

        else:
            xgb_cv_params = est.named_steps['model'].get_xgb_params().copy()
            if 'n_estimators' in xgb_cv_params:
                del xgb_cv_params['n_estimators']  # remove for early stopping

            xgb_cv_dtrain = xgb.DMatrix(data=self._X_train,
                                        label=self._y_train)
            cv_results = xgb.cv(
                dtrain=xgb_cv_dtrain, params=xgb_cv_params, nfold=3,
                num_boost_round=2000, early_stopping_rounds=50, metrics='rmse',
                as_pandas=True, seed=self.random_state)
            result_name = 'test-rmse-mean'

        opt_n_ests = cv_results[result_name].idxmin()  # note the best

        if plot:
            # plot cv errors (train and cv test)
            plt.subplot(2, 1, 1)
            plt.plot(cv_results.get('train-rmse-mean', []), label='train')
            plt.plot(cv_results[result_name], label='test', color='orange')
            plt.legend()
            plt.subplot(2, 1, 2)
            iloc_ = 100  # now exclude results with a small number of ests
            plt.plot(cv_results[result_name].iloc[iloc_:],
                     label='test', color='orange')
            plt.legend()
            plt.show()

        # update n_estimators based on the early stopping
        msg = ('updating n_estimators from {} to {} based on cv early stopping'
               .format(est.named_steps['model'].n_estimators, opt_n_ests))
        self.logger.info(msg)

        est.named_steps['model'].set_params(n_estimators=opt_n_ests)
        est.fit(X=self._X_train, y=self._y_train)  # refit
        return est

    def model_with_xgb_api(self):
        self.logger.info('modelling with xgboost api...')
        self.logger.warning('not developed like model_with_sklearn_api() '
                            'method')

        # use xgboot's api
        params = {
            "objective": "reg:linear",
            'colsample_bytree': 0.3,
            'learning_rate': 0.1,
            'max_depth': 5,
            'alpha': 10,
        }
        cv_results = xgb.cv(dtrain=self._data_dmatrix, params=params, nfold=3,
                            num_boost_round=50, early_stopping_rounds=10,
                            metrics="rmse", as_pandas=True,
                            seed=self.random_state)  # for given hyper-params
        self.logger.info(cv_results.head())  # view
        # note: at time of writing at least (2018-12), it seems that xgboost
        # does not provide its own api functionality for optimising
        # hyper-parameters, presumably because that is covered by the
        # sklearn api

        # examine the final boosting round metric
        self.logger.info((cv_results["test-rmse-mean"]).tail(1))

        xg_reg = xgb.train(params=params, dtrain=self._data_dmatrix,
                           num_boost_round=10)
        return xg_reg

    def model_with_early_stopping_cv(self, cv_routine='grid_search',
                                     param_grid=None):
        """Fit model using early stopping embedded within the cross validation
        of the hyper-parameters.
        """
        self.logger.info('modelling with early stopping in cv...')

        if param_grid is None:
            param_grid = self._get_default_param_gird(cv_routine=cv_routine,
                                                      use_lightgbm=False)

        param_sets = self._get_param_sets(cv_routine=cv_routine,
                                          param_grid=param_grid)

        cv_results_dict = self._get_cv_results(param_sets=param_sets)

        best_params = cv_results_dict['best']['params'].copy()
        best_params['n_estimators'] = cv_results_dict['best']['n_estimators']
        self.logger.info('pipeline fitted with hyper-parameters: {}'
                         .format(best_params))

        # specify model and fit using sklearn api
        xg_reg = xgb.XGBRegressor(
            objective='reg:linear', random_state=self.random_state,
            **best_params)
        xg_reg.fit(X=self._X_train, y=self._y_train)

        # get predictions for test set
        preds = xg_reg.predict(self._X_test)

        # get root mean-square-error in test set
        rmse = np.sqrt(mean_squared_error(y_true=self._y_test, y_pred=preds))
        self.logger.info('RMSE in test set: {}'.format(rmse))

        return xg_reg

    def _get_param_sets(self, cv_routine, param_grid):
        if cv_routine == 'grid_search':
            param_sets = ParameterGrid(param_grid)
            self.logger.info('number of param sets for cv: {}'
                             .format(len(param_sets)))

        elif cv_routine == 'randomised_search':
            n_iter = np.prod(
                [len(p) for p in self.default_grid_search_params.values()])
            self.logger.info('number of samples of param sets for cv: {}'
                             .format(n_iter))
            param_sets = ParameterSampler(param_grid, n_iter=n_iter,
                                          random_state=self.random_state)

        elif cv_routine == 'bayes_search':
            msg = ('bayes_search cv_routine not implemented with early '
                   'stopping in cv (n_estimators should be optimised well '
                   'by bayes_search anyway)')
            raise NotImplementedError(msg)

        else:
            raise ValueError('unrecognised cv_routine {}'.format(cv_routine))

        return param_sets

    def _get_cv_results(self, param_sets, n_cv_repeats=10):
        xgb_cv_dtrain = xgb.DMatrix(data=self._X_train, label=self._y_train)
        metric = 'rmse'  # note: change code below if greater is better!!!

        cv_results_dict = {  # initialise
            'best': {'score': None, 'n_estimators': np.nan, 'params': None},
            'full': []
        }
        for cv_params in param_sets:
            if 'n_estimators' in cv_params:
                del cv_params['n_estimators']  # remove for early stopping

            best_score_i = []  # initialise
            best_n_ests_i = []
            for i_cv_rep in range(n_cv_repeats):
                cv_results = xgb.cv(
                    dtrain=xgb_cv_dtrain, params=cv_params, nfold=3,
                    num_boost_round=2000, early_stopping_rounds=50,
                    metrics=metric, as_pandas=True, verbose_eval=False,
                    seed=self.random_state + i_cv_rep)

                # note the best score and n_estimators out of fold
                best_score_i += [
                    cv_results['test-{}-mean'.format(metric)].min()]
                best_n_ests_i += [
                    cv_results['test-{}-mean'.format(metric)].idxmin()]

            # take means as representative of results for this param set
            best_score_i = np.mean(best_score_i)
            best_n_ests_i = int(
                np.maximum(1, np.floor(np.mean(best_n_ests_i))))

            # record if best seen so far
            best_cv_score_ = cv_results_dict['best']['score']
            if best_cv_score_ is None or best_score_i < best_cv_score_:
                cv_results_dict['best']['score'] = best_score_i
                cv_results_dict['best']['n_estimators'] = best_n_ests_i
                cv_results_dict['best']['params'] = cv_params.copy()

            # record full results
            cv_results_dict['full'] += [{
                'score': best_score_i,
                'n_estimators': best_n_ests_i,
                'params': cv_params.copy(),
            }]

        return cv_results_dict


if __name__ == '__main__':
    xgd = XgboostExample()
    xgd.load_data()
    # xg_reg = xgd.model_with_sklearn_api(cv_routine='randomised_search',
    #                                     plot=False, use_lightgbm=False)
    # xg_reg = xgd.model_with_sklearn_api(cv_routine='bayes_search', plot=False)
    xg_reg = xgd.model_with_early_stopping_cv(cv_routine='randomised_search')


    ### RESULTS FROM RUNNING BAYES_SEARCH WITH 360 ITERAIONS ###
    ### NOTE: THIS TOOK OVER AN HOUR TO RUN ###
    #
    # INFO: __main__:modelling with sklearn api...
    #
    # INFO: __main__:setting random state of estimator (to 123)
    #
    # INFO: __main__:fitting pipeline with hyper-parameter candidates {
    #     'model__learning_rate': (0.05, 0.35, 'log-uniform'),
    #     'model__gamma': (1e-09, 1.0, 'log-uniform'),
    #     'model__min_child_weight': (1, 10),
    #     'model__max_delta_step': (0, 10),
    #     'model__max_depth': (2, 8),
    #     'model__colsample_bytree': (0.2, 1.0, 'uniform'),
    #     'model__colsample_bylevel': (0.2, 1.0, 'uniform'),
    #     'model__subsample': (0.2, 1.0, 'uniform'),
    #     'model__reg_alpha': (1e-09, 100.0, 'log-uniform'),
    #     'model__reg_lambda': (1e-09, 100.0, 'log-uniform'),
    #     'model__n_estimators': (20, 1000)
    # }
    #
    # INFO: __main__:pipeline fitted with hyper-parameters: {
    #     'model__colsample_bylevel': 0.4440770849236859,
    #     'model__colsample_bytree': 0.739170926232301,
    #     'model__gamma': 0.024480012859050642,
    #     'model__learning_rate': 0.05,
    #     'model__max_delta_step': 4,
    #     'model__max_depth': 4,
    #     'model__min_child_weight': 7,
    #     'model__n_estimators': 625,
    #     'model__reg_alpha': 0.13615773278411772,
    #     'model__reg_lambda': 1e-09,
    #     'model__subsample': 1.0
    # }
    #
    # INFO: __main__:RMSE in test set: 3.688307008683497
