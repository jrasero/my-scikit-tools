import numpy as np
from time import time

from sklearn import datasets
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from my_sklearn_tools.model_selection import StratifiedKFoldReg
from my_sklearn_tools.pca_regressors import LassoPCR, LogisticPCR


def test_lasso_pcr():
    """compare LassoPCR with the same using GridSearchCV"""

    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, y_train = X[:300], y[:300]
    X_test, y_test = X[300:], y[300:]

    pip = make_pipeline(VarianceThreshold(),
                        StandardScaler(with_std=False),
                        PCA(),
                        Lasso(random_state=0, max_iter=10000)
                        )

    n_alphas = 100
    alphas = np.logspace(-4, 0., n_alphas)
    tuned_parameters = [{'lasso__alpha': alphas}]

    cv = StratifiedKFoldReg(n_splits=5, random_state=0, shuffle=True)

    clf = GridSearchCV(pip,
                       tuned_parameters,
                       cv=cv,
                       scoring='neg_mean_squared_error')
    t_0 = time()
    clf.fit(X_train, y_train)
    t_f = time()

    time_1 = t_f-t_0
    y_pred_1 = clf.predict(X_test)
    score_1 = clf.score(X_test, y_test)
    alpha_opt_1 = clf.best_params_['lasso__alpha']

    lasso_pcr = LassoPCR(cv=cv, alphas=alphas, lasso_kws={'random_state': 0,
                                                          'max_iter': 10000})
    t_0 = time()
    lasso_pcr.fit(X_train, y_train)
    t_f = time()

    time_2 = t_f-t_0
    y_pred_2 = lasso_pcr.predict(X_test)
    score_2 = lasso_pcr.score(X_test, y_test)
    alpha_opt_2 = lasso_pcr.alpha_

    assert np.allclose(y_pred_1, y_pred_2)
    assert score_1 == score_2
    assert alpha_opt_1 == alpha_opt_2

    print("time using GridSearchCV: %f s,"
          " time using LassoPCR object: %f s" % (time_1, time_2))
    assert time_1 > time_2


def test_logistic_pcr():
    """compare logisticPCR with the same using GridSearchCV"""

    X, y = datasets.load_breast_cancer(return_X_y=True)
    X_train, y_train = X[:50], y[:50]
    X_test, y_test = X[50:], y[50:]

    pip = make_pipeline(VarianceThreshold(),
                        StandardScaler(with_std=False),
                        PCA(n_components=5),
                        LogisticRegression(random_state=0, max_iter=10000)
                        )

    n_cs = 10
    Cs = np.logspace(-4, 4., n_cs)
    tuned_parameters = [{'logisticregression__C': Cs}]

    cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)

    clf = GridSearchCV(pip,
                       tuned_parameters,
                       cv=cv,
                       scoring='balanced_accuracy')
    t_0 = time()
    clf.fit(X_train, y_train)
    t_f = time()

    time_1 = t_f-t_0
    y_pred_1 = clf.predict(X_test)
    score_1 = clf.score(X_test, y_test)
    c_opt_1 = clf.best_params_['logisticregression__C']

    log_pcr = LogisticPCR(cv=cv, Cs=Cs,
                          pca_kws={'n_components': 5},
                          logistic_kws={'random_state': 0, 'max_iter': 10000}
                          )
    t_0 = time()
    log_pcr.fit(X_train, y_train)
    t_f = time()

    time_2 = t_f-t_0
    y_pred_2 = log_pcr.predict(X_test)
    score_2 = log_pcr.score(X_test, y_test)
    c_opt_2 = log_pcr.C_

    assert np.allclose(y_pred_1, y_pred_2)
    assert score_1 == score_2
    assert c_opt_1 == c_opt_2

    print("time using GridSearchCV: %f s,"
          " time using LogisticPCR object: %f s" % (time_1, time_2))
    assert time_1 > time_2


def test_penaly_logistic_pcr():
    """test that penalty in logisticPCR works"""

    X, y = datasets.make_classification(n_samples=50,
                                        n_features=100,
                                        n_informative=10,
                                        n_redundant=20)

    log_pcr = LogisticPCR(cv=5, Cs=10,
                          pca_kws={'n_components': 30},
                          logistic_kws={'random_state': 0, 'max_iter': 10000,
                                        'penalty': 'l2', 'solver': 'liblinear'}
                          )

    log_pcr.fit(X, y)
    components_l2 = log_pcr.components_
    non_zero_coefs_l2 = sum(abs(log_pcr.coef_[0, :]) > 0)

    log_pcr = LogisticPCR(cv=5, Cs=10,
                          pca_kws={'n_components': 30},
                          logistic_kws={'random_state': 0, 'max_iter': 10000,
                                        'penalty': 'l1', 'solver': 'liblinear'}
                          )

    log_pcr.fit(X, y)
    components_l1 = log_pcr.components_
    non_zero_coefs_l1 = sum(abs(log_pcr.coef_[0, :]) > 0)

    # The PCA components should be the same no matter which penalty used
    assert np.allclose(components_l2, components_l1)
    assert non_zero_coefs_l2 == 30  # The number of components, no shrinkage
    assert non_zero_coefs_l1 < non_zero_coefs_l2
