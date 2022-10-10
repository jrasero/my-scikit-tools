import numpy as np

from collections.abc import Iterable
import numbers

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection._split import _CVIterableWrapper, _RepeatedSplits

__all__ = ['StratifiedKFoldReg', 'RepeatedStratifiedKFoldReg', 'check_cv']


class StratifiedKFoldReg(StratifiedKFold):

    """
    This class generate cross-validation partitions
    for regression setups, such that these partitions
    resemble the original sample distribution of the
    target variable.
    """

    def split(self, X, y, groups=None):

        n_samples = len(y)

        # Number of labels to discretize our target variable,
        # into bins of quasi equal size
        n_labels = int(np.floor(n_samples/self.n_splits))

        # Assign a label to each bin of n_splits points
        y_labels_sorted = np.concatenate([np.repeat(ii, self.n_splits)
                                          for ii in range(n_labels)])

        # Get number of points that would fall
        # out of the equally-sized bins
        mod = np.mod(n_samples, self.n_splits)

        # Find unique idxs of first unique label's ocurrence
        _, labels_idx = np.unique(y_labels_sorted, return_index=True)

        # sample randomly the label idxs to which assign the
        # the mod points
        rnd = check_random_state(self.random_state)
        rand_label_ix = rnd.choice(labels_idx, mod, replace=False)

        # insert these at the beginning of the corresponding bin
        y_labels_sorted = np.insert(y_labels_sorted,
                                    rand_label_ix,
                                    y_labels_sorted[rand_label_ix])

        # find each element of y to which label corresponds in the sorted
        # array of labels
        map_labels_y = dict()
        for ix, label in zip(np.argsort(y), y_labels_sorted):
            map_labels_y[ix] = label

        # put labels according to the given y order then
        y_labels = np.array([map_labels_y[ii] for ii in range(n_samples)])

        return super().split(X, y_labels, groups)


def check_cv(cv=5, y=None, *, classifier=False):
    """Input checker utility for building a cross-validator
    Parameters
    ----------
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if classifier is True and ``y`` is either
        binary or multiclass, :class:`StratifiedKFold` is used. In all other
        cases, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        .. versionchanged:: 0.22
            ``cv`` default value changed from 3-fold to 5-fold.
    y : array-like, default=None
        The target variable for supervised learning problems.
    classifier : bool, default=False
        Whether the task is a classification task, in which case
        stratified KFold will be used.
    Returns
    -------
    checked_cv : a cross-validator instance.
        The return value is a cross-validator which generates the train/test
        splits via the ``split`` method.
    """
    cv = 5 if cv is None else cv
    if isinstance(cv, numbers.Integral):
        if (classifier and (y is not None) and
                (type_of_target(y) in ('binary', 'multiclass'))):
            return StratifiedKFold(cv)
        else:
            return StratifiedKFoldReg(cv)

    if not hasattr(cv, 'split') or isinstance(cv, str):
        if not isinstance(cv, Iterable) or isinstance(cv, str):
            raise ValueError("Expected cv as an integer, cross-validation "
                             "object (from sklearn.model_selection) "
                             "or an iterable. Got %s." % cv)
        return _CVIterableWrapper(cv)

    return cv  # New style cv objects are passed without any modification


class RepeatedStratifiedKFoldReg(_RepeatedSplits):
    """Repeated Stratified K-Fold cross validator for Regression problems.
    Repeats Stratified K-Fold n times with different randomization in each
    repetition.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.
    random_state : int, RandomState instance or None, default=None
        Controls the generation of the random states for each repetition.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    
    
    Notes
    -----
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.
  
    """

    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(
            StratifiedKFoldReg,
            n_repeats=n_repeats,
            random_state=random_state,
            n_splits=n_splits,
        )