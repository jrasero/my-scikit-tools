#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 23:06:51 2022

@author: javi
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import check_cv
from sklearn.utils import check_X_y
from sklearn.ensemble._base import _fit_single_estimator
from sklearn.model_selection import cross_val_predict
from sklearn.base import TransformerMixin, is_classifier, is_regressor, clone
from sklearn.utils.validation import check_is_fitted

from my_sklearn_tools.model_selection import check_cv


class ColPredTransform(TransformerMixin):
    """Columnwise transformer with predictions. 
    
    Each column is replaced with the cross-validated predictions by means of
    a linear regression.
    
    Parameters
    ----------
    
    cv : int, cross-validation generator or iterable, default=None
       Determines the cross-validation splitting strategy.
       Possible inputs for cv are:
           
       - None, to use the default 5-fold cross-validation,
       - int, to specify the number of folds.
       - :term:`CV splitter`,
       - An iterable yielding (train, test) splits as arrays of indices.
       For int/None inputs, :class:`KFold` is used.
       Refer :ref:`User Guide <cross_validation>` for the various
       cross-validation strategies that can be used here.

           
   verbose : bool or int, default=False
       Amount of verbosity.
       
   n_jobs : int, default=None
       Number of CPUs to use during the cross validation.
       ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
       ``-1`` means using all processors. """

    
    def __init__(self, cv=None, n_jobs=-1, verbose=0):
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y):

        # Checkings here
       X, y = check_X_y(X, y)

       self.n_features_ = X.shape[1]
       
       fitted_estims = Parallel(n_jobs=self.n_jobs)(
           delayed(_fit_single_estimator)(clone(LinearRegression()),
                                          x[:,None],
                                          y) for x in X.T
           )
       self.estimators_ = fitted_estims
       
       return self

    def transform(self, X):

        check_is_fitted(self)

        preds = [
            getattr(est, "pred")(x[:, None])
            for x, est in zip(X.T, self.estimators_)
            ]
        return np.column_stack(preds)

    def fit_transform(self, X, y):

       self.fit(X, y)

       cv = check_cv(self.cv, y, classifier=False)
       
       preds = Parallel(n_jobs=self.n_jobs)(
           delayed(cross_val_predict)(LinearRegression(),
                                      x[:, None],
                                      y,
                                      cv=cv,
                                      n_jobs=self.n_jobs,
                                      verbose=self.verbose) 
           for x in X.T)
       
       return  np.column_stack(preds)
