#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zach Yao
@license: None
@contact: yaoliangchn@qq.com
@software: Pycharm Community.
@file: lateFusion.py
@time: 2023/6/2 20:08
@desc:
'''
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

clf = SVC(kernel='rbf')
clf.fit(train_feature,train_label)

print("OK")
