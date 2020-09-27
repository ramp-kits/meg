import functools
import glob
import numpy as np
import os
import pandas as pd
from scipy import sparse

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_score

from sklearn.base import is_classifier
from rampwf.prediction_types.base import BasePrediction
from rampwf.workflows import SKLearnPipeline
from rampwf.score_types import BaseScoreType
import warnings


DATA_HOME = "data"
RANDOM_STATE = 42

# Author: Maria Telenczuk <github: maikia>
# License: BSD 3 clause


# define the scores
class HammingLoss(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='hamming loss', precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_proba, y_proba):
        score = 100 * hamming_loss(y_true_proba, y_proba)
        return score


class JaccardError(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='jaccard error', precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_proba, y_proba):
        score = 1 - jaccard_score(y_true_proba, y_proba, average='samples')
        return score


class _MultiOutputClassification(BasePrediction):
    def __init__(self, n_columns, y_pred=None, y_true=None, n_samples=None):
        self.n_columns = n_columns
        if y_pred is not None:
            self.y_pred = np.array(y_pred)
        elif y_true is not None:
            self.y_pred = np.array(y_true)
        elif n_samples is not None:
            if self.n_columns == 0:
                shape = (n_samples)
            else:
                shape = (n_samples, self.n_columns)
            self.y_pred = np.empty(shape, dtype=float)
            self.y_pred.fill(np.nan)
        else:
            raise ValueError(
                'Missing init argument: y_pred, y_true, or n_samples')
        self.check_y_pred_dimensions()

    @classmethod
    def combine(cls, predictions_list, index_list=None):
        """Inherits from the base class where the scores are averaged.
        Here, averaged predictions < 0.5 will be set to 0.0 and averaged
        predictions >= 0.5 will be set to 1.0 so that `y_pred` will consist
        only of 0.0s and 1.0s.
        """
        # call the combine from the BasePrediction
        combined_predictions = super(
            _MultiOutputClassification, cls
            ).combine(
                predictions_list=predictions_list,
                index_list=index_list
                )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            combined_predictions.y_pred[
                combined_predictions.y_pred < 0.5] = 0.0
            combined_predictions.y_pred[
                combined_predictions.y_pred >= 0.5] = 1.0

        return combined_predictions


# Workflow for the classification problem which uses predict instead of
# predict_proba
class EstimatorMEG(SKLearnPipeline):
    """Choose predict method.

    Parameters
    ----------
    predict_method : {'auto', 'predict', 'predict_proba',
            'decision_function'}, default='auto'
        Prediction method to use. If 'auto', uses 'predict_proba' when
        estimator is a classifier and 'predict' otherwise.
    """
    def __init__(self, predict_method='auto'):
        super().__init__()
        self.predict_method = predict_method

    def test_submission(self, estimator_fitted, X):
        """Predict using a fitted estimator.

        Parameters
        ----------
        estimator_fitted : Estimator object
            A fitted scikit-learn estimator.
        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features)
            The test data set.

        Returns
        -------
        pred : ndarray of shape (n_samples, n_classes) or (n_samples)
        """
        methods = ('auto', 'predict', 'predict_proba', 'decision_function')

        if self.predict_method not in methods:
            raise NotImplementedError(f"'method' should be one of: {methods} "
                                      f"Got: {self.predict_method}")
        if self.predict_method == 'auto':
            if is_classifier(estimator_fitted):
                return estimator_fitted.predict_proba(X)
            return estimator_fitted.predict(X)
        elif hasattr(estimator_fitted, self.predict_method):
            # call estimator with the `predict_method`
            est_predict = getattr(estimator_fitted, self.predict_method)
            return est_predict(X)
        else:
            raise NotImplementedError("Estimator does not support method: "
                                      f"{self.predict_method}.")


def make_workflow():
    # defines new workflow, where predict instead of predict_proba is called
    return EstimatorMEG(predict_method='predict')


def partial_multioutput(cls=_MultiOutputClassification, **kwds):
    # this class partially inititates _MultiOutputClassification with given
    # keywords
    class _PartialMultiOutputClassification(_MultiOutputClassification):
        __init__ = functools.partialmethod(cls.__init__, **kwds)
    return _PartialMultiOutputClassification


def make_multioutput(n_columns):
    return partial_multioutput(n_columns=n_columns)


problem_title = 'Source localization of MEG signal'
n_parcels = 450  # number of parcels in each brain used in this challenge
Predictions = make_multioutput(n_columns=n_parcels)
workflow = make_workflow()

score_types = [
    HammingLoss(name='hamming loss (%)'),
    JaccardError(name='jaccard error')  # TODO: decide on the score
]


def get_cv(X, y):
    test = os.getenv('RAMP_TEST_MODE', 0)
    n_splits = 8
    if test:
        n_splits = 2
    cv = ShuffleSplit(n_splits=n_splits, test_size=0.5,
                      random_state=RANDOM_STATE)
    return cv.split(X, y)


def _read_data(path, dir_name):
    X_df = pd.read_csv(os.path.join(DATA_HOME, dir_name, 'X.csv.gz'))
    X_df.iloc[:, :-1] *= 1e12  # scale data to avoid tiny numbers
    y = sparse.load_npz(
        os.path.join(DATA_HOME, dir_name, 'target.npz')).toarray()
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        # First 2 subjects
        X_df = X_df.iloc[:3000, :]
        y = y[:3000, :]
        # Every 20th sample
        X_df = X_df.iloc[::20, :]
        y = y[::20, :]
        return X_df, y
    else:
        return X_df, y


def get_train_data(path="."):
    return _read_data(path, 'train')


def get_test_data(path="."):
    return _read_data(path, 'test')


def get_leadfields():
    data_dir = 'data/'

    # find all the files ending with '_lead_field' in the data directory
    lead_field_files = os.path.join(data_dir, '*lead_field.npz')
    lead_field_files = sorted(glob.glob(lead_field_files))

    parcel_indices, Ls = {}, {}

    for lead_file in lead_field_files:
        lead_field = np.load(lead_file)
        lead_file = os.path.basename(lead_file)
        subject_id = 'subject_' + lead_file.split('_')[1]
        parcel_indices[subject_id] = lead_field['parcel_indices']
        # scale L to avoid tiny numbers
        Ls[subject_id] = 1e8 * lead_field['lead_field']
        assert parcel_indices[subject_id].shape[0] == Ls[subject_id].shape[1]

    assert len(parcel_indices) == len(Ls)
    assert len(parcel_indices) >= 1  # at least a single subject

    return Ls, parcel_indices
