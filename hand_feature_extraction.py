#!/usr/bin/env python
# encoding: utf-8
'''
@author: Yawen Xu
@license: None
@contact: yaoliangchn@qq.com
@software: Pycharm Community.
@file: data_post_process_lungBlast.py
@time: 2023/6/2 9:45
@desc:
'''

import glob
import os
import cv2
import numpy as np
import skimage
from skimage.feature import local_binary_pattern
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from hand_feature_utils import glcm_feature_extract, lbp_feature_extract, fft_feature_extract, sift_to_vector_extract, \
    mean_bright_extract, mean_bright_gradient_extract
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib


def read_and_extract_features(video_files_train):
    my_dict={}
    all_feature_list = []
    label = []
    for folder in os.listdir(video_files_train):
        print(folder)
        mean_bright = []
        mean_bright_gradient = []
        glcm_feature = []
        lbp_feature = []
        fft_feature = []
        sift_to_vector = []
        # glcm_feature_list = []
        # lbp_feature_list = []
        # fft_feature_list = []
        # sift_to_vector_list = []
        # mean_bright_list = []
        # mean_bright_gradient_list = []
        folder_list = os.listdir(os.path.join(video_files_train, folder))
        folder_sorted = sorted(folder_list, key=lambda x: int(x.split('.')[0]))

        # print(folder_list)
        # print(folder_sorted)

        for filename in folder_sorted[:3]:
            print(filename)
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = cv2.imread(os.path.join(video_files_train, folder, filename), cv2.IMREAD_GRAYSCALE)

            mean_bright.append(mean_bright_extract(img))
            mean_bright_gradient.append(mean_bright_gradient_extract(img))
            glcm_feature.append(glcm_feature_extract(img))
            lbp_feature.append(lbp_feature_extract(img))
            fft_feature.append(fft_feature_extract(img))
            sift_to_vector.append(sift_to_vector_extract(img))
            # print("读取了一张图片")

        mean_bright_list = np.max(mean_bright)
        mean_bright_gradient_list = np.max(mean_bright_gradient)
        glcm_feature_list = np.max(glcm_feature, axis=0)
        lbp_feature_list = np.max(lbp_feature, axis=0)
        fft_feature_list = np.max(fft_feature, axis=0)
        sift_to_vector_list = np.max(sift_to_vector, axis=0)

        glcm_feature_list = np.array(glcm_feature_list).flatten()
        lbp_feature_list = np.array(lbp_feature_list).flatten()
        fft_feature_list = np.array(fft_feature_list).flatten()
        sift_to_vector_list = np.array(sift_to_vector_list).flatten()
        label = folder.split('.')[0][-1]

        if (label == "a"):
            label = [0, 1]
        if (label == "b"):
            label = [1, 0]
        print(label)

        arr2 = np.asarray([mean_bright_list, mean_bright_gradient_list]).flatten()

        all_feature2 = np.concatenate((fft_feature_list, glcm_feature_list), axis=0)
        all_feature3 = np.concatenate((sift_to_vector_list, lbp_feature_list, label), axis=0)
        all_feature = all_feature3
        my_dict[folder] = all_feature


        all_feature_list.append(all_feature)

    return all_feature_list, my_dict


def train_and_test_svm(features_train, labels_train, features_test, labels_test):
    # 训练 SVM 分类器
    print("begin SVM")
    class_weight = {0: 78., 1: 23.}
    svm_classifier = SVC(kernel='poly', class_weight=class_weight)
    svm_classifier.fit(features_train, labels_train)
    # 测试 SVM 分类器
    y_pred = svm_classifier.predict(features_test)

    y_pred_train = svm_classifier.predict(features_train)
    c_m = confusion_matrix(labels_test, y_pred)
    print("cm", c_m)
    c_m_train = confusion_matrix(labels_train, y_pred_train)
    print('cm_train', c_m_train)
    # tsne = TSNE(n_components=3, perplexity=15.0, early_exaggeration=12.0, learning_rate=200.0,
    #             n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean',
    #             init='random', verbose=1, random_state=42)
    #
    # tsne_data = tsne.fit_transform(test_features)
    # plt.figure(figsize=(10, 10))
    # plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=y_pred, cmap=matplotlib.Spectral)
    # plt.colorbar()
    # plt.show()
    return accuracy_score(labels_test, y_pred)

def train_and_test_randomForest(features_train, labels_train, features_test, labels_test):
    print("begin random forest")
    rfc = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=None, class_weight="balanced")
    rfc.fit(features_train, labels_train)

    y_pred = rfc.predict(features_test)
    y_pred_train = rfc.predict(features_train)
    c_m_test = confusion_matrix(labels_test, y_pred)
    print("cm_test", c_m_test)
    c_m_train = confusion_matrix(labels_train, y_pred_train)
    print('cm_train', c_m_train)

    return accuracy_score(labels_test, y_pred)


def knn_clf(features_train, labels_train, features_test, labels_test):
    # 初始化 KNN 分类器
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto', metric='minkowski', p=2,
                               metric_params=None, n_jobs=None, leaf_size=30)

    # 训练模型
    knn.fit(features_train, labels_train)

    # 预测测试集
    y_pred = knn.predict(features_test)

    # 输出准确率
    acc = accuracy_score(labels_test, y_pred)
    y_pred_train = knn.predict(features_train)
    c_m_test = confusion_matrix(labels_test, y_pred)
    print("cm_test", c_m_test)
    c_m_train = confusion_matrix(labels_train, y_pred_train)
    print('cm_train', c_m_train)

    return acc


# 读取图像并提取特征
def nb_clf(features_train, labels_train, features_test, labels_test):

    # 初始化朴素贝叶斯分类器
    class_prior = [0.28, 0.72]
    nb = MultinomialNB(class_prior=class_prior, alpha=1.0, fit_prior=True)
    # 训练模型
    nb.fit(features_train, labels_train)
    # 预测测试集
    y_pred = nb.predict(features_test)
    # 输出准确率
    acc = accuracy_score(labels_test, y_pred)
    y_pred_train = nb.predict(features_train)
    c_m_test = confusion_matrix(labels_test, y_pred)
    print("cm_test", c_m_test)
    c_m_train = confusion_matrix(labels_train, y_pred_train)
    print('cm_train', c_m_train)

    print("Accuracy:", acc)

    return acc


if __name__ == "__main__":
    # data_path = "F:/Dataset/1.Medical_dataset/lungBlas_301/xyzchannel_merge/flow_before_merge"
    train_path = "F:\\Dataset\\1.Medical_dataset\\lungBlas_301\\xyzchannel_merge_split\\split_11\\train\\blast"
    # test_path = "split_xyz_v4/test"

    # train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    # my_dict = {}

    # for video_file in os.listdir(data_path):
        # label = video_file.split('.')[0][-1]
        # # print(label)
        # if (label == "a"):
        #     label = 0
        # if (label == "b"):
        #     label = 1
        # if 'a' in video_file:
        #     label = 0
        # if 'b' in video_file:
        #     lable = 1

        # train_labels.append(label)
    train_features, my_dict = read_and_extract_features(train_path)
    my_dict = np.array(my_dict, dtype=object)
    print(my_dict)
    np.save('feature_dict.npy', my_dict)
    print(train_labels)
    print("len_train", len(train_labels))

    # for video_file in os.listdir(test_path):
    #     label = video_file.split('.')[0][-1]
    #     # print(label)
    #     if (label == "a"):
    #         label = 0
    #     if (label == "b"):
    #         label = 1
    #     test_labels.append(label)
    # test_features = read_and_extract_features(test_path)
    # test_features = np.array(test_features, dtype=object)
    # print(test_features)
    # print(test_labels)
    # print("len_test", len(test_labels))
    #
    #
    # # accuracy_score = train_and_test_svm(train_features, train_labels, test_features, test_labels)
    # # accuracy_score = train_and_test_randomForest(train_features, train_labels, test_features, test_labels)
    # accuracy_score = knn_clf(train_features, train_labels, test_features, test_labels)
    # # # accuracy_score = nb_clf(train_features, train_labels, test_features, test_labels)
    #
    # print('SVM classification accuracy: {:.2f}%'.format(accuracy_score * 100))



