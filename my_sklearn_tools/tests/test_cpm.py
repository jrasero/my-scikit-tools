
"""
Test cpm stuff.
"""

import numpy as np
from sklearn.datasets import make_regression, make_classification
from my_sklearn_tools.cpm import f_correlation
from my_sklearn_tools.cpm import CPMRegression, CPMClassification


def test_f_correlation():
    """Function to test correlation."""

    from scipy.stats import pearsonr

    X, y = make_regression(n_features=20)

    corrs_np, pvals_np = np.array([pearsonr(y, x) for x in X.T]).T
    corrs_f, pvals_f = f_correlation(X, y)

    assert np.allclose(corrs_np, corrs_f)
    assert np.allclose(pvals_np, pvals_f)


def test_cpm_regression():
    """test CPM regression."""
    from sklearn.feature_selection import SelectFpr
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    X, y = make_regression(n_features=100, n_informative=20)

    signs = np.array([1 if np.random.random() < 0.5 else -1
                      for i in range(X.shape[1])])
    X = X*signs

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.7)

    sfpr = SelectFpr(score_func=f_correlation)
    linReg = LinearRegression()

    sfpr.fit(X_train, y_train)

    scores = sfpr.scores_
    pos_mask = (scores > 0) & (sfpr.get_support())
    neg_mask = (scores < 0) & (sfpr.get_support())

    pos_x_train = X_train[:, pos_mask].sum(axis=1)
    pos_x_test = X_test[:, pos_mask].sum(axis=1)

    neg_x_train = X_train[:, neg_mask].sum(axis=1)
    neg_x_test = X_test[:, neg_mask].sum(axis=1)

    # test positive model
    linReg.fit(pos_x_train.reshape(-1, 1),
               y_train)
    y_pred_1 = linReg.predict(pos_x_test.reshape(-1, 1))

    cpm = CPMRegression(strength='positive')
    cpm.fit(X_train, y_train)
    y_pred_cpm = cpm.predict(X_test)

    assert np.allclose(cpm.filter_method_.get_support(), sfpr.get_support())
    assert np.allclose(cpm.coef_, linReg.coef_)
    assert np.allclose(cpm.intercept_, linReg.intercept_)
    assert np.allclose(y_pred_cpm, y_pred_1)

    # test negative model
    linReg.fit(neg_x_train.reshape(-1, 1),
               y_train)
    y_pred_1 = linReg.predict(neg_x_test.reshape(-1, 1))

    cpm = CPMRegression(strength='negative')
    cpm.fit(X_train, y_train)
    y_pred_cpm = cpm.predict(X_test)

    assert np.allclose(cpm.filter_method_.get_support(), sfpr.get_support())
    assert np.allclose(cpm.coef_, linReg.coef_)
    assert np.allclose(cpm.intercept_, linReg.intercept_)
    assert np.allclose(y_pred_cpm, y_pred_1)

    # test both model (The default one)
    linReg.fit(np.column_stack((pos_x_train, neg_x_train)),
               y_train)
    y_pred_1 = linReg.predict(np.column_stack((pos_x_test, neg_x_test)))

    cpm = CPMRegression()
    cpm.fit(X_train, y_train)
    y_pred_cpm = cpm.predict(X_test)

    assert np.allclose(cpm.filter_method_.get_support(), sfpr.get_support())
    assert np.allclose(cpm.coef_, linReg.coef_)
    assert np.allclose(cpm.intercept_, linReg.intercept_)
    assert np.allclose(y_pred_cpm, y_pred_1)


def test_cpm_classification():
    """test CPM regression."""
    from sklearn.feature_selection import SelectFpr
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_features=100, n_informative=20)

    signs = np.array([1 if np.random.random() < 0.5 else -1
                      for i in range(X.shape[1])])
    X = X*signs

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.7)

    sfpr = SelectFpr(score_func=f_correlation)
    clf = LogisticRegression(penalty='none', class_weight='balanced')

    sfpr.fit(X_train, y_train)

    scores = sfpr.scores_
    pos_mask = (scores > 0) & (sfpr.get_support())
    neg_mask = (scores < 0) & (sfpr.get_support())

    pos_x_train = X_train[:, pos_mask].sum(axis=1)
    pos_x_test = X_test[:, pos_mask].sum(axis=1)

    neg_x_train = X_train[:, neg_mask].sum(axis=1)
    neg_x_test = X_test[:, neg_mask].sum(axis=1)

    # test positive model
    clf.fit(pos_x_train.reshape(-1, 1), y_train)
    y_pred_1 = clf.predict(pos_x_test.reshape(-1, 1))

    cpm = CPMClassification(strength='positive')
    cpm.fit(X_train, y_train)
    y_pred_cpm = cpm.predict(X_test)

    assert np.allclose(cpm.filter_method_.get_support(), sfpr.get_support())
    assert np.allclose(cpm.coef_, clf.coef_)
    assert np.allclose(cpm.intercept_, clf.intercept_)
    assert np.allclose(y_pred_cpm, y_pred_1)

    # test negative model
    clf.fit(neg_x_train.reshape(-1, 1), y_train)
    y_pred_1 = clf.predict(neg_x_test.reshape(-1, 1))

    cpm = CPMClassification(strength='negative')
    cpm.fit(X_train, y_train)
    y_pred_cpm = cpm.predict(X_test)

    assert np.allclose(cpm.filter_method_.get_support(), sfpr.get_support())
    assert np.allclose(cpm.coef_, clf.coef_)
    assert np.allclose(cpm.intercept_, clf.intercept_)
    assert np.allclose(y_pred_cpm, y_pred_1)

    # test both model (The default one)
    clf.fit(np.column_stack((pos_x_train, neg_x_train)), y_train)
    y_pred_1 = clf.predict(np.column_stack((pos_x_test, neg_x_test)))

    cpm = CPMClassification()
    cpm.fit(X_train, y_train)
    y_pred_cpm = cpm.predict(X_test)

    assert np.allclose(cpm.filter_method_.get_support(), sfpr.get_support())
    assert np.allclose(cpm.coef_, clf.coef_)
    assert np.allclose(cpm.intercept_, clf.intercept_)
    assert np.allclose(y_pred_cpm, y_pred_1)
