"""Connectivity-based predictive modelling."""

import numpy as np

from scipy.sparse import issparse
from scipy import stats

from sklearn.base import BaseEstimator
from sklearn.feature_selection import GenericUnivariateSelect
# from scipy.spatial.distance import squareform
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot, row_norms


class BaseCPM(BaseEstimator):
    """Base class to be inherited by regression and classifcaiton models."""

    def __init__(self,
                 mode='fpr',
                 param=0.05,
                 strength='both'):

        self.mode = mode
        self.param = param
        self.strength = strength

    def _strength_features(self, X):
        """Compute the strength features."""
        scores = self.filter_method_.scores_
        mask = self.filter_method_.get_support()

        pos_mask = (scores*mask) > 0
        neg_mask = (scores*mask) < 0

        pos_features = X[:, pos_mask].sum(axis=1)
        neg_features = X[:, neg_mask].sum(axis=1)

        if self.strength == 'positive':
            X_strength = pos_features.reshape(-1, 1)
        elif self.strength == 'negative':
            X_strength = neg_features.reshape(-1, 1)
        else:
            X_strength = np.column_stack((pos_features, neg_features))
        # Add intercept
        return X_strength

    def _check_mode(self):

        strength_cond = self.strength == 'positive' \
            or self.strength == 'negative' or self.strength == 'both'

        if strength_cond is False:
            raise ValueError("strength mode should be 'positive',"
                             "'negative', or 'both'.")
            
    def transform(self, X):
        """Function to transform the data to the strength features."""
        check_is_fitted(self.filter_method_)
        
        return self._strength_features(X)


class CPMRegression(BaseCPM):
    """Connecivity-based predictive modelling for regression."""

    def fit(self, X, y, sample_weight=None):
        """Fit method."""
        # Assert strength mode
        self._check_mode()

        # For now, just only with on target.
        X, y = check_X_y(X, y, multi_output=False, y_numeric=True)
        self.n_features_ = X.shape[1]
        filter_method = GenericUnivariateSelect(score_func=f_correlation,
                                                mode=self.mode,
                                                param=self.param
                                                )
        filter_method.fit(X, y)
        self.filter_method_ = filter_method

        # Matrix of strengths
        X_strength = self._strength_features(X)
        # Add intercept
        intercept = np.ones(X_strength.shape[0])
        X_strength = np.column_stack((intercept, X_strength))

        coefs, _, _, _ = np.linalg.lstsq(X_strength, y, rcond=None)

        if y.ndim < 2:
            self.intercept_ = coefs[0]
            self.coef_ = coefs[1:]
        else:
            self.intercept_ = coefs[0, :]
            self.coef_ = coefs[1:, :].T  # To swap dimensions

        return self

    def predict(self, X):
        """Predict method."""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])

        n_features = self.n_features_
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], n_features))

        X_strength = self._strength_features(X)

        scores = safe_sparse_dot(X_strength, self.coef_.T,
                                 dense_output=True) + self.intercept_

        return scores  # scores.ravel() if scores.shape[1] == 1 else scores

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination of the prediction."""
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)


class CPMClassification(BaseCPM):
    """Connecivity-based predictive modelling for classification."""

    def __init__(self,
                 mode='fpr',
                 param=0.05,
                 strength='both'):

        raise NotImplementedError


def f_correlation(X, y, *, center=True):
    """Univariate linear regression tests.

    Linear model for testing the individual effect of each of many regressors.
    This is a scoring function to be used in a feature selection procedure, not
    a free standing feature selection procedure. It is just the same function
    as f_regression, just that I have here chosen to return the correlations
    instead of the F-statistics.

    For more details, see :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    X : {array-like, sparse matrix}  shape = (n_samples, n_features)
        The set of regressors that will be tested sequentially.
    y : array of shape(n_samples).
        The data matrix
    center : bool, default=True
        If true, X and y will be centered.

    Returns
    -------
    corr : array, shape=(n_features,)
        Pearson correlation values of features.
    pval : array, shape=(n_features,)
        p-values of F-scores.

    See Also
    --------
    mutual_info_regression : Mutual information for a continuous target.
    f_classif : ANOVA F-value between label/feature for classification tasks.
    chi2 : Chi-squared stats of non-negative features for classification tasks.
    SelectKBest : Select features based on the k highest scores.
    SelectFpr : Select features based on a false positive rate test.
    SelectFdr : Select features based on an estimated false discovery rate.
    SelectFwe : Select features based on family-wise error rate.
    SelectPercentile : Select features based on percentile of the highest
        scores.
    """
    X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                     dtype=np.float64)
    n_samples = X.shape[0]

    # compute centered values
    # note that E[(x - mean(x))*(y - mean(y))] = E[x*(y - mean(y))], so we
    # need not center X
    if center:
        y = y - np.mean(y)
        if issparse(X):
            X_means = X.mean(axis=0).getA1()
        else:
            X_means = X.mean(axis=0)
        # compute the scaled standard deviations via moments
        X_norms = np.sqrt(row_norms(X.T, squared=True) -
                          n_samples * X_means ** 2)
    else:
        X_norms = row_norms(X.T)

    # compute the correlation
    corr = safe_sparse_dot(y, X)
    corr /= X_norms
    corr /= np.linalg.norm(y)

    # convert to p-value
    degrees_of_freedom = y.size - (2 if center else 1)
    F = corr ** 2 / (1 - corr ** 2) * degrees_of_freedom
    pv = stats.f.sf(F, 1, degrees_of_freedom)
    return corr, pv
