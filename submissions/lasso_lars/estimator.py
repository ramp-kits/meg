import os
import glob

import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import ClassifierMixin, RegressorMixin


def _get_coef(est):
    """Get coefficients from a fitted regression estimator."""
    if hasattr(est, 'steps'):
        return est.steps[-1][1].coef_
    return est.coef_


class SparseRegressor(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, Ls, parcel_indices, model, n_jobs=1):
        self.Ls = Ls
        self.parcel_indices = parcel_indices
        self.model = model
        self.n_jobs = n_jobs

    def fit(self, X, y):
        return self

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)

    def _run_model(self, model, L, X):
        norms = np.linalg.norm(L, axis=0)
        L = L / norms[None, :]

        est_coefs = np.empty((X.shape[0], L.shape[1]))
        for idx in range(len(X)):
            x = X.iloc[idx].values
            model.fit(L, x)
            est_coef = np.abs(_get_coef(model))
            est_coef /= norms
            est_coefs[idx] = est_coef

        return est_coefs.T

    def decision_function(self, X):
        X = X.reset_index(drop=True)

        n_parcels = np.max([np.max(s) for s in self.parcel_indices.values()])
        betas = np.empty((len(X), n_parcels))
        for subj_idx in np.unique(X['subject']):
            L_used = self.Ls[subj_idx]

            X_used = X[X['subject'] == subj_idx]
            X_used = X_used.drop('subject', axis=1)

            est_coef = self._run_model(self.model, L_used, X_used)

            beta = pd.DataFrame(
                np.abs(est_coef)
            ).groupby(self.parcel_indices[subj_idx]).max().transpose()
            betas[X['subject'] == subj_idx] = np.array(beta)
        return betas


class CustomSparseEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def fit(self, L, x):
        alpha_max = abs(L.T.dot(x)).max() / len(L)
        alpha = self.alpha * alpha_max
        lasso = linear_model.LassoLars(alpha=alpha, max_iter=3,
                                       normalize=False, fit_intercept=False)
        lasso.fit(L, x)
        self.coef_ = lasso.coef_


def get_leadfields():
    data_dir = 'data/'

    # find all the files ending with '_lead_field' in the data directory
    lead_field_files = os.path.join(data_dir, '*L.npz')
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


def get_estimator():
    Ls, parcel_indices = get_leadfields()
    custom_model = CustomSparseEstimator(alpha=0.2)
    lasso_lars_alpha = \
        SparseRegressor(Ls, parcel_indices, custom_model)

    return lasso_lars_alpha
