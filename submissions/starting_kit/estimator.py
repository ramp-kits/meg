from sklearn.compose import make_column_transformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


N_JOBS = -1


def get_estimator():

    # K-means
    clf = KNeighborsClassifier(n_neighbors=3)
    kneighbours = MultiOutputClassifier(clf, n_jobs=N_JOBS)

    preprocessor = make_column_transformer(("drop", 'subject'),
                                           remainder='passthrough')

    pipeline = Pipeline([
        ('transformer', preprocessor),
        ('classifier', kneighbours)
    ])

    return pipeline
