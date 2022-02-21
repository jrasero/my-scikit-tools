"Transmodal-stacking learnin approach"
import numpy as np
from joblib import delayed, Parallel
from sklearn.base import TransformerMixin, is_classifier, is_regressor, clone
from sklearn.ensemble._base import _fit_single_estimator
from sklearn.utils import check_consistent_length
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import (LogisticRegressionCV, RidgeClassifierCV,
                                  LassoCV, RidgeCV, ElasticNetCV,
                                  LogisticRegression, RidgeClassifier,
                                  Lasso, Ridge, ElasticNet)
from sklearn.preprocessing import LabelEncoder
from my_sklearn_tools.model_selection import check_cv
from sklearn.utils.validation import check_is_fitted


class BaseTransmodal(TransformerMixin):

    def __init__(self,
                 estimators,
                 final_estimator,
                 *,
                 cv=None,
                 stack_method="auto",
                 n_jobs=1,
                 verbose=0):

        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.stack_method = stack_method
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        # Fit first level estimators
        X_multi = self._fit_first(X, y, sample_weight)
        # Fit final estimator
        _fit_single_estimator(
            self.final_estimator_, X_multi, y,
            sample_weight=sample_weight
        )

        return self

    def _fit_first(self, X, y, sample_weight):

        X = self._validate_datasets(X)

        n_channels = len(X)
        self.n_channels_ = n_channels

        # Validate first estimators
        estimators = self._validate_estimators()

        # Validate cross-validation parameter
        cv = check_cv(self.cv, y,
                      classifier=is_classifier(estimators[0])
                      )

        # Set cv to the same to tune the estims
        for est in estimators:
            if hasattr(est, "cv"):
                est.set_params(**{'cv': cv})

        # Validate final estimator
        self._validate_final_estimator(cv)

        # Fit estimators
        estims_fitted = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_estimator)(clone(est), x, y, sample_weight)
            for est, x in zip(estimators, X)
        )

        self.estimators_ = [self._get_optimal_estimator(estim,
                                                        x,
                                                        y,
                                                        sample_weight)
                            for estim, x in zip(estims_fitted, X)
                            ]

        preds = Parallel(n_jobs=self.n_jobs)(
            delayed(cross_val_predict)(est,
                                       x,
                                       y,
                                       cv=cv,
                                       method=self.stack_method,
                                       n_jobs=self.n_jobs,
                                       verbose=self.verbose)
            for est, x in zip(self.estimators_, X)
            )

        X_multi = self._concatenate_predictions(preds)

        return X_multi

    def fit_transform(self, X, y, sample_weight=None):

        # Fit first level estimators
        X_multi = self._fit_first(X, y, sample_weight)

        # Fit final estimator
        _fit_single_estimator(
            self.final_estimator_, X_multi, y,
            sample_weight=sample_weight
        )

        return X_multi

    def _transform(self, X):
        """Concatenate and return the predictions of the estimators."""
        preds = [
            getattr(est, self.stack_method)(x)
            for x, est in zip(X, self.estimators_)
            ]
        return self._concatenate_predictions(preds)

    def _validate_estimators(self):

        if self.estimators is None:
            raise ValueError(
                "'estimators' attribute shouldn't be None, but an estimator "
                " instance or a list of estimator instances"
                )

        if isinstance(self.estimators, list) is False:
            estimators = [self.estimators] * self.n_channels_
        else:
            estimators = self.estimators
            if len(estimators) != self.n_channels_:
                raise ValueError("estimator is a list of estimators, with "
                                 f"{len(estimators)} components, which is "
                                 "different from the number of channels, "
                                 f"here equal to {self.n_channels_}"
                                 )

            if is_classifier(estimators[0]):
                same_estims = [is_classifier(estim) for estim in estimators]
            else:
                same_estims = [is_regressor(estim) for estim in estimators]

            if all(same_estims) is False:
                raise ValueError("Estimator should be all classifiers "
                                 "or regressors."
                                 )

        return estimators

    def _validate_datasets(self, X, n_channels=None):
        if isinstance(X, list) is False:
            raise ValueError("print X should be a list")

        if n_channels is not None:
            if len(X) != n_channels:
                raise ValueError(f"A list of {len(X)} was supplied, but "
                                 f"{n_channels} was expected")

        check_consistent_length(*X)
        return X

    def _concatenate_predictions(self, predictions):
        """Concatenate the predictions of each first-level estimators.

        This helper is in charge of ensuring the predictions are 2D arrays and
        it will drop one of the probability column when using probabilities
        in the binary case. Indeed, the p(y|c=0) = 1 - p(y|c=1)
        """
        X_multi = []
        for preds in predictions:
            # case where the the estimator returned a 1D array
            if preds.ndim == 1:
                X_multi.append(preds.reshape(-1, 1))
            else:
                if (
                    self.stack_method == "predict_proba"
                    and len(self.classes_) == 2
                ):
                    # Remove the first column when using probabilities in
                    # binary classification because both features are perfectly
                    # collinear.
                    X_multi.append(preds[:, 1:])
                else:
                    X_multi.append(preds)

        return np.column_stack(X_multi)

    # @abstractmethod
    def _validate_final_estimator(self, cv):
        pass

    def _get_optimal_estimator(self, estimator, X, y, sample_weight):
        pass


class TransmodalClassifer(BaseTransmodal):

    def __init__(self,
                 estimators,
                 final_estimator=None,
                 *,
                 cv=None,
                 stack_method="predict_proba",
                 n_jobs=1,
                 verbose=0):
        """Transmodal Classifier.

        A transmodal classifier consists in stacking the output from
        a list of datasets (first stage) and feed these predictions to a
        new classifier to yield the final prediction (second stage)
        It allows to exploit the strength of each individual dataset
        by using their output as input of a final classifier.

        Parameters
        ----------
        estimators : estimator or list of estimators.
            Estimators that fit each piece of input data. There should be as
            many estimators as datasets. If only one estimator is supplied,
            then that is fitted to all datasets.
        final_estimator : estimator, default=None
            A classifier which will combine the first-level estimators.
            If none, a LogisticRegressionCV will be used
        cv : int, cross-validation generator or an iterable, default=None
            Determines the cross-validation strategy used to generate the
            out-of-sample predictions in the training set. This cv is also
            used to tune the first-level estimators, in case they accept
            such argument (e.g. GridSearchCV).
        stack_method : {'predict_proba', 'decision_function', 'predict'}.
            default="predict_proba".
            What features to used in the stacking stage.
        n_jobs : int, default=None
            The number of jobs to run in parallel.
        verbose : int, default=0
            Verbosity level.

        """

        super().__init__(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            stack_method=stack_method,
            n_jobs=n_jobs,
            verbose=verbose,
            )

    def fit(self, X, y, sample_weight=None):
        """Fit the estimators.

        Parameters
        ----------
        X : list of {arrays-like, sparse matrix} of
            shapes (n_samples, m_i), where is the number of samples and
            m_i is the number of features for the dataset i.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,) or default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        Returns
        -------
        self : object
        """

        y_type = type_of_target(y)

        if y_type != "binary":
            raise ValueError("outcome should be (for now) "
                             "binary"
                             )

        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_

        return super().fit(X, y, sample_weight)

    def fit_transform(self, X, y, sample_weight=None):
        """Fit the estimators.

        Parameters
        ----------
        X : list of {arrays-like, sparse matrix} of
            shapes (n_samples, m_i), where is the number of samples and
            m_i is the number of features for the dataset i.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,) or default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        Returns
        -------
        self : object
        """

        y_type = type_of_target(y)

        if y_type != "binary":
            raise ValueError("outcome should be (for now) "
                             "binary"
                             )

        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_

        return super().fit_transform(X, y, sample_weight)

    def predict(self, X):
        """Predict target for a list of datasets X.

         Parameters
         ----------
         X : list of {arrays-like, sparse matrix} of
             shapes (n_samples, m_i), where is the number of samples and
             m_i is the number of features for the dataset i.
         Returns
         -------
         y_pred : ndarray of shape (n_samples,) or (n_samples, n_output)
             Predicted targets.
         """

        check_is_fitted(self)

        X = self._validate_datasets(X, self.n_channels_)

        X_multi = self._transform(X)

        return self.final_estimator_.predict(X_multi)

    def transform(self, X):
        """Concatenate and return the predictions of the estimators."""

        check_is_fitted(self)

        X = self._validate_datasets(X, self.n_channels_)

        return self._transform(X)

    def _validate_final_estimator(self, cv):

        if self.final_estimator is not None:
            self.final_estimator_ = clone(self.final_estimator)
        else:
            default = LogisticRegressionCV(penalty="l1",
                                           solver="liblinear",
                                           cv=cv)
            self.final_estimator_ = clone(default)

        if not is_classifier(self.final_estimator_):
            raise ValueError(
                "'final_estimator' parameter should be a classifier. "
                "Got {}".format(self.final_estimator_)
                )

    def _get_optimal_estimator(self, estim, X, y, sample_weight):
        "Gets optimal estimator in case of passing GridSearchCV or "
        "LogisticRegressionCV objects"

        if hasattr(estim, "best_estimator_"):
            estimator = getattr(estim, "best_estimator_")
        elif isinstance(estim, (LogisticRegressionCV, RidgeClassifierCV)):
            if isinstance(estim, LogisticRegressionCV):
                estimator = LogisticRegression(C=estim.C_[0],
                                               l1_ratio=estim.l1_ratio_[0]
                                               )
            elif isinstance(estim, RidgeClassifierCV):
                estimator = RidgeClassifier(alpha=estim.alpha_)

            # Set the rest of parameters
            params = estim.get_params()
            for key, value in params.items():
                if key in estimator.get_params().keys():
                    estimator.set_params(**{key: value})

            # Return fitted, that is, with coef and intercept
            estimator.coef_ = estim.coef_
            estimator.intercept_ = estim.intercept_
            estimator.classes_ = estim.classes_
        else:
            estimator = estim
        return estimator


class TransmodalRegressor(BaseTransmodal):

    def __init__(self,
                 estimators,
                 final_estimator=None,
                 *,
                 cv=None,
                 n_jobs=1,
                 verbose=0):
        """Transmodal Regressor.

        A transmodal regressor consists in stacking the output from
        a list of datasets (first stage) and feed these predictions to a
        new regressor to yield the final prediction (second stage)
        It allows to exploit the strength of each individual dataset
        by using their output as input of a final regressor.

        Parameters
        ----------
        estimators : estimator or list of estimators.
            Estimators that fit each piece of input data. There should be as
            many estimators as datasets. If only one estimator is supplied,
            then that is fitted to all datasets.
        final_estimator : estimator, default=None
            A regressor which will combine the first-level estimators.
            If none, a LassoCV will be used
        cv : int, cross-validation generator or an iterable, default=None
            Determines the cross-validation strategy used to generate the
            out-of-sample predictions in the training set. This cv is also
            used to tune the first-level estimators, in case they accept
            such argument (e.g. GridSearchCV).
        n_jobs : int, default=None
            The number of jobs to run in parallel.
        verbose : int, default=0
            Verbosity level.

        """

        super().__init__(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            stack_method="predict",
            n_jobs=n_jobs,
            verbose=verbose,
            )

    def fit(self, X, y, sample_weight=None):
        """Fit the estimators.

        Parameters
        ----------
        X : list of {arrays-like, sparse matrix} of
            shapes (n_samples, m_i), where is the number of samples and
            m_i is the number of features for the dataset i.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,) or default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        Returns
        -------
        self : object
        """

        y_type = type_of_target(y)

        if y_type != "continuous":
            raise ValueError("outcome should be (for now) "
                             "continuous"
                             )

        return super().fit(X, y, sample_weight)

    def predict(self, X):
        """Predict target for a list of datasets X.

         Parameters
         ----------
         X : list of {arrays-like, sparse matrix} of
             shapes (n_samples, m_i), where is the number of samples and
             m_i is the number of features for the dataset i.
         Returns
         -------
         y_pred : ndarray of shape (n_samples,) or (n_samples, n_output)
             Predicted targets.
         """
        check_is_fitted(self)

        X = self._validate_datasets(X, self.n_channels_)

        X_multi = self._transform(X)

        return self.final_estimator_.predict(X_multi)

    def transform(self, X):
        """Concatenate and return the predictions of the estimators."""

        check_is_fitted(self)

        X = self._validate_datasets(X, self.n_channels_)

        return self._transform(X)

    def _validate_final_estimator(self, cv):

        if self.final_estimator is not None:
            self.final_estimator_ = clone(self.final_estimator)
        else:
            default = LassoCV(cv=cv)
            self.final_estimator_ = clone(default)

        if not is_regressor(self.final_estimator_):
            raise ValueError(
                "'final_estimator' parameter should be a regressor. "
                "Got {}".format(self.final_estimator_)
                )

    def _get_optimal_estimator(self, estim, X, y, sample_weight):
        "Gets optimal estimator in case of passing GridSearchCV or "
        "LassoCV, RidgeCV or ElasticNetCV objects"

        if hasattr(estim, "best_estimator_"):
            estimator = getattr(estim, "best_estimator_")
        elif isinstance(estim, (LassoCV, RidgeCV, ElasticNetCV)):
            if isinstance(estim, LassoCV):
                estimator = Lasso(alpha=estim.alpha_)
                # We have to add this, because "auto" does not exist in Lasso
                estim.set_params(**{'precompute': False})
            elif isinstance(estim, RidgeCV):
                estimator = Ridge(alpha=estim.alpha_)
            elif isinstance(estim, ElasticNetCV):
                estimator = ElasticNet(alpha=estim.alpha_,
                                       l1_ratio=estim.l1_ratio_)
                # The same as with Lasso
                estim.set_params(**{'precompute': False})

            params = estim.get_params()
            for key, value in params.items():
                if key in estimator.get_params().keys():
                    estimator.set_params(**{key: value})

            # Return fitted, that is, with coef and intercept
            estimator.coef_ = estim.coef_
            estimator.intercept_ = estim.intercept_
        else:
            estimator = estim

        return estimator
