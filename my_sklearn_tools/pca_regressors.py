import numpy as np
from joblib import Memory
from tempfile import mkdtemp
from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.linear_model._coordinate_descent import _alpha_grid
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import get_scorer

from .model_selection import check_cv

__all__ = ['PCARegressionCV', 'PCARegression',
           'LassoPCR', 'RidgePCR', 'ElasticnetPCR']


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

        V = self.best_estimator_.named_steps['pca'].components_
        beta = self.best_estimator_.steps[-1][1].coef_
        insert_features = self.best_estimator_.\
            named_steps['variancethreshold'].inverse_transform
        if beta.ndim == 1:
            beta = beta[None, :]
        w = beta @ V
        # Insert discarded voxels
        w = insert_features(w)
        if w.shape[0] == 0:
            w = w[0, :]
        return w

    def _get_pca(self):

        vt = VarianceThreshold()
        ss = StandardScaler(with_std=self.scale)

        if self.pca_kws is None:
            pca = PCA()
        else:
            pca = PCA(**self.pca_kws)
        self.pca_kws = pca.get_params()

        return make_pipeline(vt, ss, pca)


class LassoPCR(BasePCR):

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

    def fit(self, X, y):

        # Checkings here
        X, y = check_X_y(X, y)
        cv = check_cv(self.cv, y, classifier=False)
        splits = list(cv.split(X, y,))

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
            alphas = _alpha_grid(X_transf, y, Xy=None,
                                 fit_intercept=False, eps=self.eps,
                                 n_alphas=self.n_alphas,
                                 normalize=False, copy_X=False)
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
        self.components_ = pip_opt.named_steps['pca'].components_
        self.coef_ = pip_opt.named_steps['lasso'].coef_

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

    def fit(self, X, y):

        # Checkings here
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        cv = check_cv(self.cv, y, classifier=True)
        splits = list(cv.split(X, y,))

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
        # Save weights in feature space
        self.weights_ = self.get_weights()

        return self


class RidgePCR(BasePCR):

    def __init__(self):
        raise NotImplementedError


class ElasticnetPCR(BasePCR):

    def __init__(self):
        raise NotImplementedError


def _cv_optimize(cv_estims,
                 X,
                 y,
                 train,
                 val,
                 transf,
                 score):

    X_train, X_val = X[train], X[val]
    y_train, y_val = y[train], y[val]

    X_train_trans = transf.fit_transform(X_train)
    X_val_trans = transf.transform(X_val)

    # Generate list of estimators to fit
    estimators_fit = [estim.fit(X_train_trans, y_train)
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
