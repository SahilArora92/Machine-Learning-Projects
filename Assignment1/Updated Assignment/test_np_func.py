import numpy as np
from utils import euclidean_distance, gaussian_kernel_distance, inner_product_distance, cosine_sim_distance
from utils import f1_score, model_selection_without_normalization, model_selection_with_transformation
from hw1_knn import KNN
from data import data_processing
from typing import List
from utils import NormalizationScaler, MinMaxScaler
import data
import hw1_dt as decision_tree
import utils as Utils
from sklearn.metrics import accuracy_score


def test_tree():
    features, labels = data.sample_decision_tree_data()
    # build the tree
    dTree = decision_tree.DecisionTree()
    dTree.train(features, labels)
    # print
    Utils.print_tree(dTree)

    # data
    X_test, y_test = data.sample_decision_tree_test()
    # testing
    y_est_test = dTree.predict(X_test)
    test_accu = accuracy_score(y_est_test, y_test)
    print('test_accu', test_accu)


def test_big_tree():
    # load data
    X_train, y_train, X_test, y_test = data.load_decision_tree_data()

    # set classifier
    dTree = decision_tree.DecisionTree()

    # training
    dTree.train(X_train.tolist(), y_train.tolist())

    # print
    Utils.print_tree(dTree)

    # testing
    y_est_test = dTree.predict(X_test)
    test_accu = accuracy_score(y_est_test, y_test)
    print('test_accu', test_accu)

scaling_classes = {
    'min_max_scale': MinMaxScaler,
    'normalize': NormalizationScaler,
}
p1_array = [0, 1, 0, 0, 1, 0]
p2_array = [0, 1, 1, 0, 0, 1]
test_array = [0, 2, 1, 0, 0, 1, 1, 1, 1]
input_feature = [[3, 4], [1, -1], [0, 0]]
input_feature2 = [[2, -1], [-1, 5], [0, 0]]
call1_feature = [[0, 10], [2, 0]]
call2_feature = [[20, 1]]
call3_feature = [[1, 1], [0, 0]]
distance_funcs = {
    'euclidean': euclidean_distance,
    'gaussian': gaussian_kernel_distance,
    'inner_prod': inner_product_distance,
    'cosine_dist': cosine_sim_distance,
}


def test_arange():
    a = np.arange(4).reshape(2, 2)
    print(a)
    print(np.diag(a))
    print(np.einsum('i...i', a))
    print(euclidean_distance(p1_array, p2_array))
    print(f1_score(p1_array, p2_array))

    # p1 = np.array(p1_array, dtype=np.float64)
    # print(np.einsum('ij,ij->i', a, a))
    # print((a ** 2).sum())


def test_normalization_scalar(features: List[List[float]]) -> List[List[float]]:
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


def test_min_max_scalar(features: List[List[float]]) -> List[List[float]]:
    """
    normalize the feature vector for each sample . For example,
    if the input features = [[2, -1], [-1, 5], [0, 0]],
    the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
    """
    min_max_features = []
    transposed_features = [list(i) for i in zip(*features)]
    for feature in transposed_features:
        min_in_feature = min(feature)
        max_in_feature = max(feature)
        if min_in_feature == max_in_feature:
            min_max_features.append([0 for x in feature])
        else:
            min_max_features.append([(x - min_in_feature) / (max_in_feature - min_in_feature) for x in feature])
    return [list(i) for i in zip(*min_max_features)]


def test_most_freq():
    knn = KNN(5, 'euclidean_distance')
    print(knn.most_frequent(test_array))


def test_model_selection():
    Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing()
    model_selection_without_normalization(distance_funcs, Xtrain.tolist(), ytrain.tolist(), Xval.tolist(),
                                          yval.tolist())


def test_transform_model_selection():
    Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing()
    model_selection_with_transformation(distance_funcs,scaling_classes, Xtrain.tolist(), ytrain.tolist(), Xval.tolist(),
                                          yval.tolist())


if __name__ == "__main__":
    # test_arange()
    # test_most_freq()
    # print(test_model_selection())
    # print(test_transform_model_selection())
    # print(test_normalization_scalar(input_feature))

    # print(test_min_max_scalar(input_feature2))
    # obj = MinMaxScaler()
    # print(obj(call1_feature))
    # print(obj(call2_feature))
    test_big_tree()
    #test_tree()
