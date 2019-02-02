from __future__ import division, print_function

from typing import List

import numpy as np
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    # TODO: save features and lable to self
    def train(self, features: List[List[float]], labels: List[int]):
        # features: List[List[float]] a list of points
        # labels: List[int] labels of features
        self.feat = features
        self.labels = labels

    # TODO: predict labels of a list of points
    def predict(self, features: List[List[float]]) -> List[int]:
        # features: List[List[float]] a list of points
        # return: List[int] a list of predicted labels
        majority_labels = []
        for test_feat_point in features:
            majority_labels.append(self.most_frequent(self.get_k_neighbors(test_feat_point)))
        return majority_labels

    def most_frequent(self, labels: List[int]):

        hash_dict = dict()
        for i in range(len(labels)):
            if labels[i] in hash_dict.keys():
                hash_dict[labels[i]] += 1
            else:
                hash_dict[labels[i]] = 1

        # find the max frequency
        max_count = 0
        res = -1
        for i in hash_dict:
            if max_count < hash_dict[i]:
                res = i
                max_count = hash_dict[i]
        return res

    # TODO: find KNN of one point
    def get_k_neighbors(self, point: List[float]) -> List[int]:
        # point: List[float] one example
        # return: List[int] labels of K nearest neighbor
        dis = []
        for curr_feat, curr_label in zip(self.feat, self.labels):
            dis.append((self.distance_function(curr_feat, point), curr_label))
        dis.sort()
        k_near_neighbor = [x[1] for x in dis]
        return k_near_neighbor[:self.k]


if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
