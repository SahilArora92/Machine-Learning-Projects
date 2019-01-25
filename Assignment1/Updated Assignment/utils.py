import numpy as np
from typing import List
from hw1_knn import KNN


# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    raise NotImplementedError


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    raise NotImplementedError


# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t',
                       deep=deep + 1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')


# TODO: implement F1 score
def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    assert len(real_labels) == len(predicted_labels)
    tp, fp, fn = 0, 0, 0
    for real, pred in zip(real_labels, predicted_labels):
        if real == 1 and pred == 1:
            tp += 1
        elif real == 0 and pred == 1:
            fn += 1
        elif real == 1 and pred == 0:
            fp += 1

    if tp > 0:
        precision = float(tp) / (tp + fp)
        recall = float(tp) / (tp + fn)
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0.0


# TODO:
def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    p1 = np.array(point1, dtype=np.float64)
    p2 = np.array(point2, dtype=np.float64)
    return np.linalg.norm(p1 - p2)


# TODO:
def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    p1 = np.array(point1, dtype=np.float64)
    p2 = np.array(point2, dtype=np.float64)
    return np.inner(p1, p2)


# TODO:
def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    p1 = np.array(point1, dtype=np.float64)
    p2 = np.array(point2, dtype=np.float64)
    return -np.exp(-1 / 2 * (np.linalg.norm(p1 - p2) ** 2))


# TODO:
def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    p1 = np.array(point1, dtype=np.float64)
    p2 = np.array(point2, dtype=np.float64)
    sum_yy = (p2 ** 2).sum()
    sum_xx = (p1 ** 2).sum()
    sum_xy = p1.dot(p2.T)
    return 1 - ((sum_xy / np.sqrt(sum_xx)) / np.sqrt(sum_yy))


# TODO: select an instance of KNN with the best f1 score on validation dataset
def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model

    upper_bound = len(Xtrain)
    if upper_bound > 30:
        upper_bound = 30

    max_f1 = []
    for key, distance_func in distance_funcs.items():
        max_score = -1
        min_k = 0
        best_model = []
        for k in range(1, upper_bound, 2):
            knn = KNN(k, distance_func)
            knn.train(Xtrain, ytrain)
            pred_labels = knn.predict(Xval)
            curr_f1 = f1_score(yval, pred_labels)
            if curr_f1 > max_score:
                max_score = curr_f1
                min_k = k
                best_model = knn
        max_f1.append((max_score, key, min_k, best_model))
    max_f1.sort(reverse=True)

    # filter ties
    majority = filter_ties(max_f1)

    SORT_ORDER = {"euclidean": 0, "gaussian": 1, "inner_prod": 2, "cosine_dist": 3}
    # break ties
    majority.sort(key=lambda val: SORT_ORDER[val[1]])

    return majority[0][3], majority[0][2], majority[0][1]


# TODO: select an instance of KNN with the best f1 score on validation dataset, with normalized data
def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # scaling_classes: diction of scalers
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    # return best_scaler: best function choosed for best_model

    upper_bound = len(Xtrain)
    if upper_bound > 30:
        upper_bound = 30
    max_f1 = []
    for scale_class_name, scaling_class in scaling_classes.items():
        scale_obj = scaling_class()
        if scale_class_name == 'normalize':
            trans_Xtrain = transpose_list(scale_obj(transpose_list(Xtrain)))
            trans_Xval = transpose_list(scale_obj(transpose_list(Xval)))
        else:
            trans_Xtrain = scale_obj(Xtrain)
            trans_Xval = scale_obj(Xval)

        for dist_func_name, distance_func in distance_funcs.items():
            max_score = -1
            min_k = 0
            best_model = []
            for k in range(1, upper_bound, 2):
                knn = KNN(k, distance_func)
                knn.train(trans_Xtrain, ytrain)
                pred_labels = knn.predict(trans_Xval)
                curr_f1 = f1_score(yval, pred_labels)
                if curr_f1 > max_score:
                    max_score = curr_f1
                    min_k = k
                    best_model = knn
            max_f1.append((max_score, scale_class_name, dist_func_name, min_k, best_model))
    max_f1.sort(reverse=True)

    # filter ties
    majority = filter_ties(max_f1)

    # break ties
    SORT_ORDER_SCALAR = {"min_max_scale": 0, "normalize": 1}
    majority.sort(key=lambda val: SORT_ORDER_SCALAR[val[1]])

    majority = filter_ties(max_f1)

    SORT_ORDER = {"euclidean": 0, "gaussian": 1, "inner_prod": 2, "cosine_dist": 3}
    majority.sort(key=lambda val: SORT_ORDER[val[2]])

    return majority[0][4], majority[0][3], majority[0][2], majority[0][1]


def filter_ties(max_f1):
    majority = [max_f1[0]]
    index = 1
    for next_row in max_f1[1:]:
        if majority[0][0] == next_row[0]:
            majority.append(max_f1[index])
        index += 1
    return majority


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        normalized_features = []
        for feature in features:
            sum_squares = 0
            for i in feature:
                sum_squares += i * i
            sum_squares_root = np.sqrt(sum_squares)
            if sum_squares == 0:
                normalized_features.append(feature)
            else:
                normalized_features.append([x / sum_squares_root for x in feature])
        return normalized_features


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """

    def __init__(self):
        self.first_run = True
        self.max_val = []
        self.min_val = []

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        if self.first_run:
            min_max_features = []
            transposed_features = transpose_list(features)
            for feature in transposed_features:
                min_in_feature = min(feature)
                max_in_feature = max(feature)
                self.min_val.append(min_in_feature)
                self.max_val.append(max_in_feature)
                if min_in_feature == max_in_feature:
                    min_max_features.append([0 for x in feature])
                else:
                    min_max_features.append([(x - min_in_feature) / (max_in_feature - min_in_feature) for x in feature])
            self.first_run = False
            return transpose_list(min_max_features)
        else:
            min_max_features = []
            transposed_features = transpose_list(features)
            index = 0
            for feature in transposed_features:
                min_in_feature = self.min_val[index]
                max_in_feature = self.max_val[index]
                if min_in_feature == max_in_feature:
                    min_max_features.append([0 for x in feature])
                else:
                    min_max_features.append([(x - min_in_feature) / (max_in_feature - min_in_feature) for x in feature])
                index += 1
            return transpose_list(min_max_features)


def transpose_list(list_array):
    return [list(i) for i in zip(*list_array)]
