from sklearn.compose import make_column_transformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


N_JOBS = 1


def get_estimator():

    # K-nearest neighbors
    clf = KNeighborsClassifier(n_neighbors=3, algorithm='brute')
    kneighbors = MultiOutputClassifier(clf, n_jobs=N_JOBS)

    preprocessor = make_column_transformer(('drop', ['subject', 'L_path']),
                                           remainder='passthrough')

    pipeline = Pipeline([
        ('transformer', preprocessor),
        ('classifier', kneighbors)
    ])

    return pipeline
