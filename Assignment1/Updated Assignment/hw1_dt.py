import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        self.feature_dim = len(features[0])
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
            # print ("feature: ", feature)
            # print ("pred: ", pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        self.classes = np.unique(labels)
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    # TODO: try to split current node
    def split(self):
        if self.splittable:
            labels_split = attribute_split_count(self.labels)

            feat_transpose = Util.transpose_list(self.features)

            # calculate S
            S = Util.calc_entropy(labels_split, sum(labels_split))

            info_gain_all_features = []
            col_index = 0

            for feat_col in feat_transpose:
                # branches
                unique_feat_vals = np.unique(feat_col).tolist()
                branches = {}
                for item in unique_feat_vals:
                    branches[item] = {}
                    for cls in self.classes:
                        branches[item][cls] = 0

                for feat_val, feat_label in zip(feat_col, self.labels):
                    branches[feat_val][feat_label] += 1

                # convert branches dict to 2D array of counts only
                branches_2d_array = []
                for key, branch in branches.items():
                    temp_array = []
                    for inner_key, count in branch.items():
                        temp_array.append(count)
                    branches_2d_array.append(temp_array)

                info_gain_all_features.append(
                    (Util.Information_Gain(S, branches_2d_array), unique_feat_vals, col_index))
                col_index += 1
            info_gain_all_features.sort(key=lambda tup: tup[0], reverse=True)

            # # check for ties
            # if len(info_gain_all_features) > 1:
            #     if info_gain_all_features[0][0] > info_gain_all_features[1][0]:
            #         self.assign_selected_feature(info_gain_all_features[0])
            #     else:

            # filter ties

            if not info_gain_all_features:
                if self.dim_split is None:
                    self.splittable = False
                return

            info_gain_all_features = Util.filter_ties(info_gain_all_features)

            info_gain_all_features.sort(key=lambda tup: tup[1], reverse=True)

            info_gain_all_features = Util.filter_ties(info_gain_all_features)

            info_gain_all_features.sort(key=lambda tup: tup[2])

            self.assign_selected_feature(info_gain_all_features[0])

            # assign Children
            # The children variable is a list of TreeNode after split
            #  the current node based on the best attributes.
            self.feature_uniq_split.sort()
            for feat_val_extract in self.feature_uniq_split:
                extract_feat = []
                extract_labels = []

                for row_feat, row_labels in zip(self.features, self.labels):
                    if feat_val_extract == row_feat[self.dim_split]:
                        temp_row_feat = row_feat[:]

                        temp_row_feat.pop(self.dim_split)
                        # if not temp_row_feat and np.unique(self.labels).size > 1:
                        #     temp_row_feat.append(popped_item)
                        extract_feat.append(temp_row_feat)
                        extract_labels.append(row_labels)

                # if np.unique(extract_labels).size > 1 and is_list_empty(extract_feat):
                #     for i in range(len(extract_feat)):
                #         extract_feat[i].append(feat_val_extract)
                self.children.append(TreeNode(extract_feat, extract_labels, np.unique(extract_labels).size))
            for node in self.children:
                node.split()
        else:
            return

    def assign_selected_feature(self, next_feature):
        self.feature_uniq_split = next_feature[1]
        self.dim_split = next_feature[2]


    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        raise NotImplementedError


def attribute_split_count(labels):
    labels_split = []
    for label in np.unique(labels):
        labels_split.append(labels.count(label))
    return labels_split


def is_list_empty(inList):
    if isinstance(inList, list):  # Is a list
        return all(map(is_list_empty, inList))
    return False  # Not a list
