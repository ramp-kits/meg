import os
import glob

import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import ClassifierMixin
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline


def _get_coef(est):
    if hasattr(est, 'steps'):
        return est.steps[-1][1].coef_
    return est.coef_


class SparseRegressor(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, model, n_jobs=1):
        self.parcel_indices = []
        self.lead_field = []
        self.model = model
        self.n_jobs = n_jobs

    def fit(self, X, y):
        return self

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)

    def _run_model(self, model, L, X, fraction_alpha=0.2):
        norms = np.linalg.norm(L, axis=0)
        L = L / norms[None, :]

        est_coefs = np.empty((X.shape[0], L.shape[1]))
        for idx, idx_used in enumerate(X.index.values):
            x = X.iloc[idx].values
            model.fit(L, x)
            est_coef = np.abs(self._get_coef(model))
            est_coef /= norms
            est_coefs[idx] = est_coef

        return est_coefs.T

    def decision_function(self, X):
        X = X.copy()
        X.iloc[:, :-2] *= 1e12

        L, parcel_indices_L, subj_dict = self._get_lead_field_info()
        # use only Lead Fields of the subjects found in X
        subj_dict = dict((k, subj_dict[k]) for k in np.unique(X['subject']))
        self.lead_field, self.parcel_indices = [], []
        subj_dict_x = {}
        for idx, s_key in enumerate(subj_dict.keys()):
            subj_dict_x[s_key] = idx
            self.lead_field.append(L[subj_dict[s_key]])
            self.parcel_indices.append(parcel_indices_L[subj_dict[s_key]])

        X['subject_id'] = X['subject'].map(subj_dict_x)
        X.astype({'subject_id': 'int32'}).dtypes
        model = MultiOutputRegressor(self.model, n_jobs=self.n_jobs)
        X = X.reset_index(drop=True)

        betas = np.empty((len(X), 0)).tolist()
        for subj_idx in np.unique(X['subject_id']):
            l_used = self.lead_field[subj_idx]

            X_used = X[X['subject_id'] == subj_idx]
            X_used = X_used.iloc[:, :-2]

            norms = l_used.std(axis=0)
            l_used = l_used / norms[None, :]

            alpha_max = abs(l_used.T.dot(X_used.T)).max() / len(l_used)
            alpha = 0.2 * alpha_max
            model.estimator.alpha = alpha
            model.fit(l_used, X_used.T)  # cross validation done here

            for idx, idx_used in enumerate(X_used.index.values):
                est_coef = np.abs(_get_coef(model.estimators_[idx]))
                est_coef /= norms
                beta = pd.DataFrame(
                        np.abs(est_coef)
                        ).groupby(
                        self.parcel_indices[subj_idx]).max().transpose()
                betas[idx_used] = np.array(beta).ravel()
        betas = np.array(betas)
        return betas

    def _get_lead_field_info(self):
        data_dir = 'data/'

        lead_field_files = os.path.join(data_dir, '*lead_field.npz')
        lead_field_files = sorted(glob.glob(lead_field_files))

        parcel_indices_leadfield, L = [], []
        subj_dict = {}
        for idx, lead_file in enumerate(lead_field_files):
            lead_matrix = np.load(lead_file)

            lead_file = os.path.basename(lead_file)
            subj_dict['subject_' + lead_file.split('_')[1]] = idx

            parcel_indices_leadfield.append(lead_matrix['parcel_indices'])

            # scale L to avoid tiny numbers
            L.append(1e8 * lead_matrix['lead_field'])
            assert parcel_indices_leadfield[idx].shape[0] == L[idx].shape[1]

        assert len(parcel_indices_leadfield) == len(L) == idx + 1
        assert len(subj_dict) >= 1  # at least a single subject

        return L, parcel_indices_leadfield, subj_dict


def get_estimator():

    model_lars = linear_model.LassoLars(max_iter=3, normalize=False,
                                        fit_intercept=False)

    lasso_lars = SparseRegressor(model_lars)

    pipeline = Pipeline([
        ('classifier', lasso_lars)
    ])

    return pipeline
