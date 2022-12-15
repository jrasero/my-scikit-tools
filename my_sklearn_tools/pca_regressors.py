import numpy as np
from joblib import Memory
from tempfile import mkdtemp
from joblib import Parallel, delayed
import itertools

from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.linear_model import Lasso, LogisticRegression, Ridge, ElasticNet
from sklearn.linear_model._coordinate_descent import _alpha_grid
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import get_scorer

from .model_selection import check_cv
from .preprocessing import ColPredTransform

__all__ = ['PCARegressionCV', 'PCARegression',
           'LassoPCR', #'NewLassoPCR', 
           'RidgePCR', 'ElasticNetPCR', 'LogisticPCR']


class PCARegressionCV():

    def __init__(self,
                 scale=False,
                 cv=None,
                 n_jobs=None,
                 cache_dir=True,
                 verbose=0):

        self.scale = scale
        self.cv = cv
        self.n_jobs = n_jobs
        self.cache_dir = cache_dir
        self.verbose = verbose

    def build(self, reg_type):

        if type(reg_type) is str:
            try:
                reg = self._get_regressor(reg_type)
            except Exception:
                msg = "Regression type is not valid"
                ValueError(msg)

        if self.cache_dir:
            tmpfolder = mkdtemp()
            memory = Memory(location=tmpfolder, verbose=self.verbose)
        else:
            memory = None

        if self.scale:
            pip = make_pipeline(VarianceThreshold(), StandardScaler(), PCA(),
                                reg,
                                memory=memory)
        else:
            pip = make_pipeline(VarianceThreshold(), PCA(), reg,
                                memory=memory)

        param_grid = self._get_param_grid(reg_type)
        grid = GridSearchCV(pip,
                            param_grid,
                            cv=self.cv,
                            n_jobs=self.n_jobs,
                            scoring="neg_mean_squared_error")

        return grid

    def _get_regressor(self,
                       reg_type):

        regression_types = {'lasso': linear_model.Lasso(max_iter=1e6),
                            'ridge': linear_model.Ridge(),
                            'elasticnet':
                                linear_model.ElasticNet(max_iter=1e6),
                            # 'lars': TODO,
                            # 'lassolars': TODO
                            }
        reg = regression_types[reg_type]
        return reg

    def _get_param_grid(self, reg_type):

        alphas = 10**np.linspace(-4, 2, 100)
        l1_ratio = [0.25, 0.5, 0.75]
        param_grids = {
            'lasso': {'lasso__alpha': alphas},
            'ridge': {'ridge__alpha': alphas},
            'elasticnet': {'elasticnet__alpha': alphas,
                           'elasticnet__l1_ratio': l1_ratio},
            # 'lars': TODO,
            # 'lassolars': TODO
                            }

        param_grid = param_grids[reg_type]
        return param_grid


class PCARegression():

    def __init__(self,
                 scale=False,
                 cache_dir=False,
                 verbose=0):
        self.scale = scale
        self.cache_dir = cache_dir
        self.verbose = verbose

    def build(self, reg_type):

        if type(reg_type) is str:
            try:
                reg = self._get_regressor(reg_type)
            except Exception:
                msg = "Regression type is not valid"
                ValueError(msg)

        if self.cache_dir:
            tmpfolder = mkdtemp()
            memory = Memory(location=tmpfolder, verbose=self.verbose)
        else:
            memory = None

        if self.scale:
            pip = make_pipeline(VarianceThreshold(), StandardScaler(), PCA(),
                                reg,
                                memory=memory)
        else:
            pip = make_pipeline(VarianceThreshold(), PCA(), reg,
                                memory=memory)

        return pip

    def _get_regressor(self, reg_type):

        regression_types = {'ols': linear_model.LinearRegression(),
                            'lasso': linear_model.Lasso(max_iter=1e6),
                            'ridge': linear_model.Ridge(),
                            'elasticnet':
                                linear_model.ElasticNet(max_iter=1e6)
                            # 'lars': TODO,
                            # 'lassolars': TODO
                            }

        reg = regression_types[reg_type]
        return reg


class BasePCR(BaseEstimator):

    def predict(self, X):

        check_is_fitted(self)

        return self.best_estimator_.predict(X)

    def score(self, X, y):

        check_is_fitted(self)

        return self.scorer_(self.best_estimator_, X, y)

    def get_weights(self):

        check_is_fitted(self)

        # Here it's just essentially multiplying by the mixing V matrix,
        # given that we start from the demeaned data...
        pca_inv_transf = self.best_estimator_.\
            named_steps['pca'].inverse_transform
        vt_1_inv_transf = self.best_estimator_.\
            named_steps['variancethreshold-1'].inverse_transform
        vt_2_inv_transf = self.best_estimator_.\
            named_steps['variancethreshold-2'].inverse_transform

        beta = self.best_estimator_[-1].coef_

        if beta.ndim == 1:
            beta = beta[None, :]

        # We are computing the weights for the centered or scaled data,
        # that's why we don't transform with the StandardScaler step
        w = vt_1_inv_transf(pca_inv_transf(vt_2_inv_transf(beta)))

        # Return weights to original units if we had scaled the data..
        if self.scale:
            w = w/self.best_estimator_.named_steps['standardscaler'].scale_

        if w.shape[0] == 1:
            w = w[0, :]
        return w

    def _get_pca(self):

        vt_1 = VarianceThreshold()
        ss = StandardScaler(with_std=self.scale)

        if self.pca_kws is None:
            pca = PCA()
        else:
            pca = PCA(**self.pca_kws)
        self.pca_kws = pca.get_params()

        # This is to remove last PC which is usually constant and small.
        vt_2 = VarianceThreshold(threshold=1e-20)

        return make_pipeline(vt_1, ss, pca, vt_2)


class LassoPCR(BasePCR):
    """
    
    """

    def __init__(self,
                 scale=False,
                 cv=None,
                 n_alphas=100,
                 alphas=None,
                 eps=1e-3,
                 pca_kws=None,
                 lasso_kws=None,
                 scoring='neg_mean_squared_error',
                 n_jobs=None,
                 verbose=0
                 ):

        self.scale = scale
        self.cv = cv
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.eps = eps
        self.pca_kws = pca_kws
        self.lasso_kws = lasso_kws
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None, *, groups=None):

        # Checkings here
        X, y = check_X_y(X, y)
        cv = check_cv(self.cv, y, classifier=False)
        splits = list(cv.split(X, y, groups=groups))

        pip_transf = self._get_pca()

        if self.lasso_kws is None:
            lasso = Lasso()
        else:
            lasso = Lasso(**self.lasso_kws)
        lasso_kws = lasso.get_params()
        lasso_kws.pop('alpha')
        self.lasso_kws = lasso_kws

        if self.alphas is None:
            X_transf = pip_transf.fit_transform(X)
            alphas = _alpha_grid(X_transf, y,
                                 eps=self.eps,
                                 n_alphas=self.n_alphas,
                                 fit_intercept=lasso_kws['fit_intercept'],
                                 copy_X=lasso_kws['copy_X'])
        else:
            alphas = np.sort(self.alphas)[::-1]
        self.alphas_ = alphas

        estimators = [Lasso(alpha=alpha, **lasso_kws) for alpha in alphas]

        self.scorer_ = get_scorer(self.scoring)

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)

        scores_cv = parallel(
            delayed(_cv_optimize)(
                estimators.copy(), X, y, train, val,
                clone(pip_transf),
                self.scorer_
                ) for train, val in splits)
        scores_cv = np.column_stack(scores_cv)
        self.scores_cv_ = scores_cv
        scores_cv_mean = np.mean(scores_cv, axis=1)
        alpha_opt = alphas[np.argmax(scores_cv_mean)]
        self.alpha_ = alpha_opt

        lasso_opt = Lasso(alpha=alpha_opt, **lasso_kws)
        pip_opt = clone(pip_transf)
        pip_opt.steps.append(make_pipeline(lasso_opt).steps[0])
        pip_opt.fit(X, y)
        self.best_estimator_ = pip_opt

        # Save singular matrix and beta coefficients in component space
        #self.components_ = pip_opt.named_steps['pca'].components_
        self.coef_ = pip_opt.named_steps['lasso'].coef_
        self.intercept_ = pip_opt.named_steps['lasso'].intercept_

        # Save weights in feature space
        self.weights_ = self.get_weights()

        return self


class LogisticPCR(BasePCR):

    def __init__(self,
                 scale=False,
                 cv=None,
                 Cs=10,
                 pca_kws=None,
                 logistic_kws=None,
                 scoring='balanced_accuracy',
                 n_jobs=None,
                 verbose=0
                 ):

        self.scale = scale
        self.cv = cv
        self.Cs = Cs
        self.pca_kws = pca_kws
        self.logistic_kws = logistic_kws
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, *, groups=None):

        # Checkings here
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        cv = check_cv(self.cv, y, classifier=True)

        splits = list(cv.split(X, y, groups=groups))

        if isinstance(self.Cs, int):
            Cs = np.logspace(-4, 4, self.Cs)
        else:
            Cs = self.Cs
        self.Cs_ = Cs

        pip_transf = self._get_pca()

        if self.logistic_kws is None:
            clf = LogisticRegression()
        else:
            clf = LogisticRegression(**self.logistic_kws)

        logistic_kws = clf.get_params()
        logistic_kws.pop('C')
        self.logistic_kws = logistic_kws

        estimators = [LogisticRegression(C=C, **logistic_kws) for C in
                      self.Cs_]

        self.scorer_ = get_scorer(self.scoring)

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)

        scores_cv = parallel(
            delayed(_cv_optimize)(
                estimators.copy(), X, y, train, val,
                clone(pip_transf),
                self.scorer_
                ) for train, val in splits)
        scores_cv = np.column_stack(scores_cv)

        self.scores_cv_ = scores_cv
        scores_cv_mean = np.mean(scores_cv, axis=1)
        c_opt = self.Cs_[np.argmax(scores_cv_mean)]
        self.C_ = c_opt

        clf_opt = LogisticRegression(C=c_opt, **logistic_kws)
        pip_opt = clone(pip_transf)
        pip_opt.steps.append(make_pipeline(clf_opt).steps[0])
        pip_opt.fit(X, y)
        self.best_estimator_ = pip_opt

        # Save singular matrix and beta coefficients in component space
        self.components_ = pip_opt.named_steps['pca'].components_
        self.coef_ = pip_opt.named_steps['logisticregression'].coef_
        self.intercept_ = pip_opt.named_steps['logisticregression'].intercept_

        # Save weights in feature space
        self.weights_ = self.get_weights()

        return self

    def predict_proba(self, X):

        check_is_fitted(self)
        return self.best_estimator_.predict_proba(X)

    def predict_log_proba(self, X):

        check_is_fitted(self)
        return self.best_estimator_.predict_log_proba(X)


class RidgePCR(BasePCR):

    def __init__(self,
                 scale=False,
                 cv=None,
                 alphas=100,
                 pca_kws=None,
                 ridge_kws=None,
                 scoring='neg_mean_squared_error',
                 n_jobs=None,
                 verbose=0
                 ):

        self.scale = scale
        self.cv = cv
        self.alphas = alphas
        self.pca_kws = pca_kws
        self.ridge_kws = ridge_kws
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, *, groups=None):

        # Checkings here
        X, y = check_X_y(X, y)
        cv = check_cv(self.cv, y, classifier=True)

        splits = list(cv.split(X, y, groups=groups))

        if isinstance(self.alphas, int):
            alphas = np.logspace(-4, 4, self.alphas)
        else:
            alphas = self.alphas
        self.alphas_ = alphas

        pip_transf = self._get_pca()

        if self.ridge_kws is None:
            ridge = Ridge()
        else:
            ridge = Ridge(**self.ridge_kws)

        ridge_kws = ridge.get_params()
        ridge_kws.pop('alpha')
        self.ridge_kws = ridge_kws

        estimators = [Ridge(alpha=alpha, **ridge_kws) for alpha in
                      self.alphas_]

        self.scorer_ = get_scorer(self.scoring)

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)

        scores_cv = parallel(
            delayed(_cv_optimize)(
                estimators.copy(), X, y, train, val,
                clone(pip_transf),
                self.scorer_
                ) for train, val in splits)
        scores_cv = np.column_stack(scores_cv)

        self.scores_cv_ = scores_cv
        scores_cv_mean = np.mean(scores_cv, axis=1)
        alpha_opt = self.alphas_[np.argmax(scores_cv_mean)]
        self.alpha_ = alpha_opt

        ridge_opt = Ridge(alpha=alpha_opt, **ridge_kws)
        pip_opt = clone(pip_transf)
        pip_opt.steps.append(make_pipeline(ridge_opt).steps[0])
        pip_opt.fit(X, y)
        self.best_estimator_ = pip_opt

        # Save singular matrix and beta coefficients in component space
#        self.components_ = pip_opt.named_steps['pca'].components_
        self.coef_ = pip_opt.named_steps['ridge'].coef_
        self.intercept_ = pip_opt.named_steps['ridge'].intercept_

        # Save weights in feature space
        self.weights_ = self.get_weights()

        return self


class ElasticNetPCR(BasePCR):
    
    def __init__(self,
                 scale=False,
                 cv=None,
                 l1_ratio = 0.5,
                 n_alphas=100,
                 alphas=None,
                 eps=1e-3,
                 pca_kws=None,
                 elasticnet_kws=None,
                 scoring='neg_mean_squared_error',
                 n_jobs=None,
                 verbose=0
                 ):

        self.scale = scale
        self.cv = cv
        self.l1_ratio = l1_ratio
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.eps = eps
        self.pca_kws = pca_kws
        self.elasticnet_kws = elasticnet_kws
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        
    def fit(self, X, y, sample_weight=None, *, groups=None):

        # Checkings here
        X, y = check_X_y(X, y)
        cv = check_cv(self.cv, y, classifier=False)
        splits = list(cv.split(X, y, groups=groups))

        pip_transf = self._get_pca()

        if self.elasticnet_kws is None:
            elasticnet = ElasticNet()
        else:
            elasticnet = ElasticNet(**self.elasticnet_kws)
        elasticnet_kws = elasticnet.get_params()
        elasticnet_kws.pop('alpha')
        elasticnet_kws.pop('l1_ratio')
        
        self.elasticnet_kws = elasticnet_kws
        
        l1_ratios = np.atleast_1d(self.l1_ratio)
        n_l1_ratio = len(l1_ratios)
        
        if self.alphas is None:
            X_transf = pip_transf.fit_transform(X)
            
            alphas = [_alpha_grid(X_transf, y,
                                  eps=self.eps,
                                  n_alphas=self.n_alphas,
                                  l1_ratio=l1_ratio,
                                  fit_intercept=elasticnet_kws['fit_intercept'],
                                  copy_X=elasticnet_kws['copy_X'])  
                      for l1_ratio in l1_ratios]
            alphas = np.row_stack(alphas)
        else:
            alphas = np.sort(self.alphas)[::-1]
            alphas = np.tile(np.sort(alphas)[::-1], (n_l1_ratio, 1))
            
        self.alphas_ = alphas

        estimators = [[ElasticNet(alpha=alpha,l1_ratio=l1_ratio, **elasticnet_kws) 
                       for alpha in alphas[ii, :]] 
                      for ii, l1_ratio in enumerate(l1_ratios)]
        
        estimators = list(itertools.chain.from_iterable(estimators))

        self.scorer_ = get_scorer(self.scoring)

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)

        scores_cv = parallel(
            delayed(_cv_optimize)(
                estimators.copy(), X, y, train, val,
                clone(pip_transf),
                self.scorer_
                ) for train, val in splits)
        
        # reshape to each fold  results to dimensions: (n_l1_ratios, n_alphas)
        scores_cv = [np.reshape(scores, alphas.shape) for scores in scores_cv]
        scores_cv = np.stack(scores_cv, axis=-1)
        
        self.scores_cv_ = scores_cv
        
        scores_cv_mean = np.mean(scores_cv, axis=2)
        
        self.alpha_ = alphas[np.unravel_index(scores_cv_mean.argmax(), 
                                              scores_cv_mean.shape)]
        self.l1_ratio_ = l1_ratios[np.unravel_index(scores_cv_mean.argmax(),
                                                    scores_cv_mean.shape)[0]]

        elasticnet_opt = ElasticNet(alpha=self.alpha_,
                                    l1_ratio= self.l1_ratio_,
                                    **elasticnet_kws)
        pip_opt = clone(pip_transf)
        pip_opt.steps.append(make_pipeline(elasticnet_opt).steps[0])
        pip_opt.fit(X, y)
        self.best_estimator_ = pip_opt

        # Save singular matrix and beta coefficients in component space
        #self.components_ = pip_opt.named_steps['pca'].components_
        self.coef_ = pip_opt.named_steps['elasticnet'].coef_
        self.sparse_coef_ = pip_opt.named_steps['elasticnet'].sparse_coef_
        self.intercept_ = pip_opt.named_steps['elasticnet'].intercept_

        # Save weights in feature space
        self.weights_ = self.get_weights()

        return self


def _cv_optimize(cv_estims,
                 X,
                 y,
                 train,
                 val,
                 transf,
                 score):

    X_train, X_val = X[train], X[val]
    y_train, y_val = y[train], y[val]

    X_train_trans = transf.fit_transform(X_train, y_train)
    X_val_trans = transf.transform(X_val)

    # Generate list of estimators to fit
    estimators_fit = [clone(estim).fit(X_train_trans, y_train)
                      for estim in cv_estims]
    scores = [score(estim, X_val_trans, y_val)
              for estim in estimators_fit]
    return scores


def check_scoring(scoring):
    from sklearn.metrics import r2_score, mean_squared_error

    if scoring == 'neg_mean_squared_error':
        def neg_mse(y_true, y_pred):
            return -mean_squared_error(y_true, y_pred)
        score = neg_mse
    elif scoring == 'r2_score':
        score = r2_score
    else:
        ValueError("scoring must be either 'neg_mean_squared_error'"
                   "or 'r2_score")
    return score
