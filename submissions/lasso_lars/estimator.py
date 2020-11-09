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
    """ Provided regression estimator (ie model) solves inverse problem
        using data X and lead field L. The estimated coefficients (est_coef
        sometimes called z) are then used to predict which parcels are active.

        X must be of a specific structure with a column name 'subject'
        matching the order in which Ls are provided."""

    def __init__(self, model, n_jobs=1):
        # self.Ls = Ls
        # self.parcel_indices = parcel_indices
        self.model = model
        self.n_jobs = n_jobs
        self.parcel_indices = {}
        self.Ls = {}

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

        for subject_id in np.unique(X['subject']):
            # load corresponding L
            L_used = X[X['subject'] == subject_id]['L_path'].iloc[0]
            lead_field = np.load(L_used)
            self.parcel_indices[subject_id] = lead_field['parcel_indices']

            # scale L to avoid tiny numbers
            self.Ls[subject_id] = 1e8 * lead_field['lead_field']
            assert (self.parcel_indices[subject_id].shape[0] ==
                    self.Ls[subject_id].shape[1])

        n_parcels = np.max([np.max(s) for s in self.parcel_indices.values()])
        betas = np.empty((len(X), n_parcels))
        for subj_idx in np.unique(X['subject']):
            L_used = self.Ls[subj_idx]

            X_used = X[X['subject'] == subj_idx]
            X_used = X_used.drop(['subject', 'L_path'], axis=1)

            est_coef = self._run_model(self.model, L_used, X_used)

            beta = pd.DataFrame(
                np.abs(est_coef)
            ).groupby(self.parcel_indices[subj_idx]).max().transpose()
            betas[X['subject'] == subj_idx] = np.array(beta)
        return betas


class CustomSparseEstimator(BaseEstimator, RegressorMixin):
    """ Regression estimator which uses LassoLars algorithm with given alpha
        normalized for each lead field L and x. """
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def fit(self, L, x):
        alpha_max = abs(L.T.dot(x)).max() / len(L)
        alpha = self.alpha * alpha_max
        lasso = linear_model.LassoLars(alpha=alpha, max_iter=3,
                                       normalize=False, fit_intercept=False)
        lasso.fit(L, x)
        self.coef_ = lasso.coef_


def get_estimator():
    # Ls, parcel_indices = get_leadfields()
    custom_model = CustomSparseEstimator(alpha=0.2)
    lasso_lars_alpha = SparseRegressor(custom_model)

    return lasso_lars_alpha
