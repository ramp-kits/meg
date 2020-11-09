from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.compose import make_column_transformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

N_JOBS = -1


class Resampler(BaseEstimator):
    ''' Resamples X and y to every 100th sample '''
    def fit_resample(self, X, y):
        return X[::100], y[::100]  # take 1% of data


def get_estimator():
    # K-nearest neighbors
    clf = KNeighborsClassifier(n_neighbors=3)
    kneighbors = MultiOutputClassifier(clf, n_jobs=N_JOBS)
    preprocessor = make_column_transformer(('drop', ['subject', 'L_path']),
                                           remainder='passthrough')
    rus = Resampler()

    pipeline = Pipeline([
        ('downsampler', rus),
        ('transformer', preprocessor),
        ('classifier', kneighbors)

        ])
    return pipeline
