import os

import numpy as np
import mne

from numba import njit

from ot import emd2


def mesh_all_distances(points, tris):
    """Compute all pairwise distances on the mesh based on edges lengths
    using Floyd-Warshall algorithm
    """
    A = mne.surface.mesh_dist(tris, points)
    A = A.toarray()
    print('Running Floyd-Warshall algorithm')
    A[A == 0.] = 1e6
    A.flat[::len(A) + 1] = 0.
    D = floyd_warshall(A)
    return D


@njit(nogil=True, cache=True)
def floyd_warshall(dist):
    npoints = dist.shape[0]
    for k in range(npoints):
        for i in range(npoints):
            for j in range(npoints):
                # If i and j are different nodes and if
                # the paths between i and k and between
                # k and j exist, do
                # d_ikj = min(dist[i, k] + dist[k, j], dist[i, j])
                d_ikj = dist[i, k] + dist[k, j]
                if ((d_ikj != 0.) and (i != j)):
                    # See if you can't get a shorter path
                    # between i and j by interspacing
                    # k somewhere along the current
                    # path
                    if ((d_ikj < dist[i, j]) or (dist[i, j] == 0)):
                        dist[i, j] = d_ikj
    return dist


def compute_ground_metric(subject, subjects_dir, annot, grade):
    """Computes pairwise distance matrix between the parcels"""
    spacing = "ico%d" % grade
    src = mne.setup_source_space(subject, spacing=spacing,
                                 subjects_dir=subjects_dir)
    tris = src[0]["use_tris"]
    vertno = src[0]["vertno"]
    points = src[0]["rr"][vertno]
    D = mesh_all_distances(points, tris)
    n_vertices = len(vertno)
    labels = mne.read_labels_from_annot(subject, annot,
                                        subjects_dir=subjects_dir)
    labels = [label.morph(subject_to=subject, subject_from=subject,
                          grade=grade) for label in labels]
    n_parcels = len(labels)
    ground_metric = np.zeros((n_parcels, n_parcels))
    for ii, label_i in enumerate(labels):
        a = np.zeros(n_vertices)
        a[label_i.vertices] = 1
        a /= a.sum()
        for jj in range(ii + 1, n_parcels):
            b = np.zeros(n_vertices)
            b[labels[jj].vertices] = 1
            b /= b.sum()
            ground_metric[ii, jj] = emd2(a, b, D)
    ground_metric = 0.5 * (ground_metric + ground_metric.T)
    ground_metric *= 1000  # change units to mm
    return ground_metric


if __name__ == "__main__":
    subjects_dir = "~/MNE_data/MNE-Sample-data/subjects"
    subjects_dir = os.path.expanduser(subjects_dir)
    grade = 3
    annot = "aparc.a2009s"
    ground_metric = compute_ground_metric("fsaverage",
                                          subjects_dir=subjects_dir,
                                          annot=annot,
                                          grade=grade)
    np.save("data/ground_metric.npy")
