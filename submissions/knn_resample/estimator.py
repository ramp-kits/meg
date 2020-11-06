import numpy as np

from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.compose import make_column_transformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import ClassifierMixin, TransformerMixin

N_JOBS = 1


class Resampler(BaseEstimator):
    ''' Resamples X and y to 2 subjects and 500 datasamples '''
    def fit_resample(self, X, y):
        X = X.reset_index(drop=True)  # make sure the indices are ordered
        # 1. take only first 2 subjects
        subjects_used = np.unique(X['subject'])[:2]
        X = X.loc[X['subject'].isin(subjects_used)]

        # 2. take only n_samples from each of the subjects
        n_samples = 500  # use only 500 samples/subject
        X = X.groupby('subject').apply(
            lambda s: s.sample(n_samples, random_state=42))

        X = X.reset_index(level=0, drop=True)  # drop subject index

        # get y corresponding to chosen X_df
        y = y[X.index]
        return X, y


class Checker(BaseEstimator, ClassifierMixin, TransformerMixin):
    def fit(self, X, y):
        print('fit')
        print(f'x.shape:{X.shape}')
        print(f'y.shape:{y.shape}')
        return self

    def transform(self, X):
        print('transform')
        print(f'x.shape:{X.shape}')
        return X


def get_estimator():
    # K-means
    clf = KNeighborsClassifier(n_neighbors=3)
    kneighbors = MultiOutputClassifier(clf, n_jobs=N_JOBS)
    preprocessor = make_column_transformer(('drop', 'subject'),
                                           remainder='passthrough')
    rus = Resampler()

    checker = Checker()
    pipeline = Pipeline([
        ('downsampler', rus),
        ('checker', checker),
        ('transformer', preprocessor),
        # ('checker2', checker),
        ('classifier', kneighbors)

        ])
    return pipeline
