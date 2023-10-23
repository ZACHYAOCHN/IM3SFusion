#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zach Yao
@license: None
@contact: yaoliangchn@qq.com
@software: Pycharm Community.
@file: lateFusion_BLS.py
@time: 2023/5/23 9:19
@desc:
'''
import argparse
import numpy as np
from bls.BroadLearningSystem import BLS, BLS_AddEnhanceNodes, BLS_AddFeatureEnhanceNodes

import numpy as np
from sklearn.svm import SVC

handfeature = np.load("./features_hand/feature_dict.npy", allow_pickle=True).item()

train_data = np.load("./tb_logs_SwinT_fuseoptical_temp/3DSwinT/train_result.npy", allow_pickle=True).item()
vali_data = np.load("./tb_logs_SwinT_fuseoptical_temp/3DSwinT/val_result.npy", allow_pickle=True).item()
test_data = np.load("./tb_logs_SwinT_fuseoptical_temp/3DSwinT/test_result.npy", allow_pickle=True).item()

# train_feature = np.zeros(shape=(len(train_data),len(handfeature['10a'])))
# train_label = np.zeros(shape=(len(train_data),2))
# train_data = np.zeros(shape=(len(train_data),2))

train_feature, train_label, train_result = [], [], []
for key in train_data.keys():
    train_feature.append(handfeature[key].tolist())
    if 'a' in key:
        train_label.append([0])
    if 'b' in key:
        train_label.append([1])
    train_result.append(train_data[key].tolist()[0])

vali_feature, vali_label, vali_result = [], [], []
for key in vali_data.keys():
    vali_feature.append(handfeature[key].tolist())
    if 'a' in key:
        vali_label.append([0])
    if 'b' in key:
        vali_label.append([1])
    vali_result.append(vali_data[key].tolist()[0])

test_feature, test_label, test_result = [], [], []
for key in test_data.keys():
    test_feature.append(handfeature[key].tolist())
    if 'a' in key:
        test_label.append([0])
    if 'b' in key:
        test_label.append([1])
    test_result.append(test_data[key].tolist()[0])

def OnehotEncoderInv(targets_matrix):
    Target = np.zeros((len(targets_matrix), 1))
    # targets = np.random.randint(0, 10, 1000)
    i = 0
    for x in range(len(targets_matrix)):
        # x = int(x)
        for i in range(len(targets_matrix[1])):
            if targets_matrix[x][i] == 1:
                Target[x] = i
            #
            # Target[i][x] = 1
            # i = i + 1
    return Target


def main(hparams):
        bls(train_data, test_data )
        # svm_test(traindata,trainlabel, testdata, testlabel)


def svm_test(traindata, trainlabel, testdata, testlabel):
    """
    SVM testing
    """
    from sklearn.svm import SVC
    from sklearn import metrics
    clf = SVC(kernel='rbf')
    trainlabel = OnehotEncoderInv(trainlabel)
    testlabel = OnehotEncoderInv(testlabel)
    clf.fit(traindata, trainlabel)
    test_label_pred = clf.predict(testdata)
    print("Acc:", metrics.accuracy_score(testlabel, test_label_pred))


def bls(traindata, testdata):
    #
    N1 = 5  # # of nodes belong to each window
    N2 = 5  # # of windows -------Feature mapping layer
    N3 = 5  # # of enhancement nodes -----Enhance layer
    L = 20  # # of incremental steps
    M1 = 50  # # of adding enhance nodes
    s = 0.8  # shrink coefficient
    C = 2 ** -30  # Regularization coefficient

    print('-------------------BLS_BASE---------------------------')
    BLS(traindata, testdata, s, C, N1, N2, N3)
    print('-------------------BLS_ENHANCE------------------------')
    BLS_AddEnhanceNodes(traindata, testdata, s, C, N1, N2, N3, L, M1)
    print('-------------------BLS_FEATURE&ENHANCE----------------')
    M2 = 5  # # of adding feature mapping nodes
    M3 = 5  # # of adding enhance nodes
    BLS_AddFeatureEnhanceNodes(traindata, testdata, s, C, N1, N2, N3, L, M1, M2, M3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='attention_subnet')
    hparams = parser.parse_args()
    main(hparams)
