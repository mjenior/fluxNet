#!/usr/bin/env python

# Import dependencies
import numpy
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from skbio.stats.distance import DistanceMatrix, permanova


#
def random(data, grouping, size=0.2, alg="manhattan", cutoff=999):
    pval = 0
    current = 0
    while pval <= 0.05:
        X_train, X_test, y_train, y_test = train_test_split(
            data, grouping, test_size=size, random_state=42, stratify=grouping
        )
        reordered = list(X_train.index) + list(X_test.index)
        groups = list(numpy.repeat(0, X_train.shape[0])) + list(numpy.repeat(1, X_test.shape[0]))
        distance_matrix = pairwise_distances(data.reindex(reordered), metric=alg)
        pval = permanova(DistanceMatrix(distance_matrix), groups, permutations=cutoff)["p-value"]
        current += 1

        if current == cutoff:
            print("WARNING: No matching training/test set distributions found")
            break

    return X_train, X_test, y_train, y_test
