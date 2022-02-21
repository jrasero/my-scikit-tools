#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 16:46:06 2022

@author: javi
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification

from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_predict,
                                     GridSearchCV)
from sklearn.linear_model import (LinearRegression, LassoCV, Lasso,
                                  LogisticRegression, LogisticRegressionCV
                                  )
from sklearn.ensemble import RandomForestClassifier

from my_sklearn_tools.transmodal import (TransmodalClassifer,
                                         TransmodalRegressor
                                         )

import pytest


def test_transmodal_reg():

    X, y = make_regression(n_samples=100,
                           n_features=10,
                           random_state=1234
                           )

    X_1, X_2 = X[:, :5], X[:, 5:]

    cv = KFold(n_splits=5, shuffle=True, random_state=1234)

    first_estimator = LinearRegression()
    trans_reg = TransmodalRegressor(first_estimator, cv=cv)

    trans_reg.fit([X_1, X_2], y)
    X_multi = trans_reg.transform([X_1, X_2])

    # Test first estimator against first peace of data
    first_estimator.fit(X_1, y)
    assert np.allclose(first_estimator.coef_,
                       trans_reg.estimators_[0].coef_)
    assert np.allclose(first_estimator.intercept_,
                       trans_reg.estimators_[0].intercept_)
    assert np.allclose(X_multi[:, 0], first_estimator.predict(X_1))

    # Test first estimator against second peace of data
    first_estimator.fit(X_2, y)
    assert np.allclose(first_estimator.coef_,
                       trans_reg.estimators_[1].coef_)
    assert np.allclose(first_estimator.intercept_,
                       trans_reg.estimators_[1].intercept_)
    assert np.allclose(X_multi[:, 1], first_estimator.predict(X_2))

    # Test training set from out-of-sample predictions
    X_multi = trans_reg.fit_transform([X_1, X_2], y)

    assert np.allclose(X_multi[:, 0],
                       cross_val_predict(first_estimator, X_1, y, cv=cv)
                       )

    assert np.allclose(X_multi[:, 1],
                       cross_val_predict(first_estimator, X_2, y, cv=cv)
                       )

    # Test second estimator
    lassocv = LassoCV(cv=cv)
    lassocv.fit(X_multi, y)
    assert np.allclose(lassocv.coef_, trans_reg.final_estimator_.coef_)

    # Test having different first-level estimators
    trans_reg = TransmodalRegressor([LassoCV(), LinearRegression()],
                                    cv=cv)
    X_multi = trans_reg.fit_transform([X_1, X_2], y)

    lassocv.fit(X_1, y)
    assert np.allclose(X_multi[:, 0],
                       cross_val_predict(Lasso(alpha=lassocv.alpha_),
                                         X_1,
                                         y,
                                         cv=cv)
                       )

    assert np.allclose(X_multi[:, 1],
                       cross_val_predict(LinearRegression(),
                                         X_2,
                                         y,
                                         cv=cv)
                       )

    # Test errors
    with pytest.raises(ValueError) as err:
        trans_reg.fit([X_1, X_2], y)
        trans_reg.predict([X_1])
    assert err.type == ValueError

    with pytest.raises(ValueError) as err:
        trans_reg.fit([X_1, X_2], y)
        trans_reg.transform([X_1])
    assert err.type == ValueError

    # Test inconsistent number of observations between datasets
    with pytest.raises(ValueError) as err:
        trans_reg.fit([X_1, X_2[:99, :]], y)
    assert err.type == ValueError


def test_transmodal_clf():

    X, y = make_classification(n_samples=100,
                               n_features=10,
                               random_state=1234
                               )

    X_1, X_2 = X[:, :5], X[:, 5:]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

    first_estimator = LogisticRegression()
    final_estimator = RandomForestClassifier(random_state=12345)

    for stack_method in ["predict_proba", "decision_function", "predict"]:
        trans_clf = TransmodalClassifer(first_estimator,
                                        final_estimator=final_estimator,
                                        cv=cv,
                                        stack_method=stack_method
                                        )

        trans_clf.fit([X_1, X_2], y)
        X_multi = trans_clf.transform([X_1, X_2])

        # Test first estimator against first peace of data
        first_estimator.fit(X_1, y)
        assert np.allclose(first_estimator.coef_,
                           trans_clf.estimators_[0].coef_)
        assert np.allclose(first_estimator.intercept_,
                           trans_clf.estimators_[0].intercept_)

        preds_first = getattr(first_estimator, stack_method)(X_1)
        if stack_method == "predict_proba":
            preds_first = preds_first[:, 1]
        assert np.allclose(X_multi[:, 0], preds_first)

        # Test first estimator against second peace of data
        first_estimator.fit(X_2, y)
        assert np.allclose(first_estimator.coef_,
                           trans_clf.estimators_[1].coef_)
        assert np.allclose(first_estimator.intercept_,
                           trans_clf.estimators_[1].intercept_)

        preds_first = getattr(first_estimator, stack_method)(X_2)
        if stack_method == "predict_proba":
            preds_first = preds_first[:, 1]
        assert np.allclose(X_multi[:, 1], preds_first)

        # Test training set from out-of-sample predictions
        X_multi = trans_clf.fit_transform([X_1, X_2], y)

        preds_first = cross_val_predict(first_estimator,
                                        X_1,
                                        y,
                                        cv=cv,
                                        method=stack_method)
        if stack_method == "predict_proba":
            preds_first = preds_first[:, 1]

        assert np.allclose(X_multi[:, 0], preds_first)

        preds_first = cross_val_predict(first_estimator,
                                        X_2,
                                        y,
                                        cv=cv,
                                        method=stack_method)
        if stack_method == "predict_proba":
            preds_first = preds_first[:, 1]

        assert np.allclose(X_multi[:, 1], preds_first)

        # Test second estimator
        final_estimator.fit(X_multi, y)
        assert np.allclose(final_estimator.feature_importances_,
                           trans_clf.final_estimator_.feature_importances_
                           )

    # Test having different first-level estimators
    trans_clf = TransmodalClassifer([LogisticRegressionCV(max_iter=1e6),
                                     LogisticRegression()],
                                    cv=cv)
    X_train_multi = trans_clf.fit_transform([X_1, X_2], y)
    X_test_multi = trans_clf.transform([X_1, X_2])

    logistic_cv = LogisticRegressionCV(cv=cv, max_iter=1e6)
    logistic_cv.fit(X_1, y)
    # Test training set predictions
    assert np.allclose(X_train_multi[:, 0],
                       cross_val_predict(LogisticRegression(C=logistic_cv.C_[0]
                                                            ),
                                         X_1,
                                         y,
                                         cv=cv,
                                         method="predict_proba")[:, 1]
                       )

    assert np.allclose(X_train_multi[:, 1],
                       cross_val_predict(LogisticRegression(),
                                         X_2,
                                         y,
                                         cv=cv,
                                         method="predict_proba")[:, 1]
                       )

    assert np.allclose(trans_clf.final_estimator_.predict(X_test_multi),
                       trans_clf.predict([X_1, X_2])
                       )

    # Test errors
    with pytest.raises(ValueError) as err:
        trans_clf.fit([X_1, X_2], y)
        trans_clf.predict([X_1])
    assert err.type == ValueError

    with pytest.raises(ValueError) as err:
        trans_clf.fit([X_1, X_2], y)
        trans_clf.transform([X_1])
    assert err.type == ValueError

    # Test inconsistent number of observations between datasets
    with pytest.raises(ValueError) as err:
        trans_clf.fit([X_1, X_2[:99, :]], y)
    assert err.type == ValueError

    # Test with gridsearchcv as first-level classifiers
    grid_rf = GridSearchCV(RandomForestClassifier(random_state=1234),
                           param_grid={'max_depth': [None, 1]}
                           )
    first_estimator = [grid_rf, LogisticRegression()]
    last_estimator = LogisticRegressionCV(cv=cv,
                                          penalty='l1',
                                          solver='liblinear',
                                          random_state=123
                                          )
    trans_clf = TransmodalClassifer(estimators=first_estimator,
                                    final_estimator=last_estimator,
                                    cv=cv)
    X_train_multi = trans_clf.fit_transform([X_1, X_2], y)
    X_test_multi = trans_clf.transform([X_1, X_2])

    grid_rf.fit(X_1, y)

    assert (trans_clf.estimators_[0].max_depth ==
            grid_rf.best_estimator_.max_depth
            )

    assert np.allclose(trans_clf.estimators_[0].feature_importances_,
                       grid_rf.best_estimator_.feature_importances_
                       )

    # Test training sets
    assert np.allclose(X_train_multi[:, 0],
                       cross_val_predict(
                           RandomForestClassifier(max_depth=1,
                                                  random_state=1234
                                                  ),
                           X_1,
                           y,
                           cv=cv,
                           method="predict_proba"
                           )[:, 1]
                       )

    # Test training sets
    assert np.allclose(X_train_multi[:, 1],
                       cross_val_predict(
                           LogisticRegression(),
                           X_2,
                           y,
                           cv=cv,
                           method="predict_proba"
                           )[:, 1]
                       )

    # Test test set on Random Forest part
    assert np.allclose(X_test_multi[:, 0], grid_rf.predict_proba(X_1)[:, 1])

    last_estimator.fit(X_train_multi, y)

    assert np.allclose(last_estimator.coef_,
                       trans_clf.final_estimator_.coef_
                       )
    assert np.allclose(last_estimator.intercept_,
                       trans_clf.final_estimator_.intercept_
                       )
    assert np.allclose(last_estimator.predict(X_test_multi),
                       trans_clf.predict([X_1, X_2])
                       )
