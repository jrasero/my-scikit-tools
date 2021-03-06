import numpy as np
from time import time

from sklearn import datasets
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score

from my_sklearn_tools.model_selection import StratifiedKFoldReg
from my_sklearn_tools.pca_regressors import LassoPCR


def test_lasso_pcr():

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
    score_1 = r2_score(y_test, y_pred_1)
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
