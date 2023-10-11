#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zach Yao
@license: None
@contact: yaoliangchn@qq.com
@software: Pycharm Community.
@file: us_op_fusion.py
@time: 2023/6/27 12:36
@desc: Fuse US with optical flow.
'''
import os
import numpy as np
import cv2

frames_path = "H:/Dataset/1.Medical_dataset/lungBlas_301_new/frames"
opFlow_path = "H:/Dataset/1.Medical_dataset/lungBlas_301_new/opticalFlow"
fuse_path = "H:/Dataset/1.Medical_dataset/lungBlas_301_new/fused256_3"

os.listdir(frames_path)

for class_dix in np.arange(len(os.listdir(frames_path))):
    data = os.path.join(frames_path, os.listdir(frames_path)[class_dix])
    op_data = os.path.join(opFlow_path, os.listdir(opFlow_path)[class_dix])

    for patient_idx in np.arange(len(os.listdir(data))):
        frames = os.path.join(data, os.listdir(data)[patient_idx])
        opflows = os.path.join(op_data, os.listdir(op_data)[patient_idx])

        for img_idx in np.arange(len(os.listdir(frames)) - 1):
            frame = cv2.imread(os.path.join(frames, os.listdir(frames)[img_idx + 1]), cv2.IMREAD_GRAYSCALE)
            frame = cv2.resize(frame, (256, 256))
            op_x = cv2.imread(os.path.join(opflows, os.listdir(opflows)[2 * img_idx]), cv2.IMREAD_GRAYSCALE)
            op_x = cv2.resize(op_x, (256, 256))
            op_y = cv2.imread(os.path.join(opflows, os.listdir(opflows)[2 * img_idx + 1]), cv2.IMREAD_GRAYSCALE)
            op_y = cv2.resize(op_y, (256, 256))

            rgb_image = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

            # Assign the grayscale images to the corresponding channels of the RGB image
            # rgb_image[:, :, 0] = frame  # Red channel
            # rgb_image[:, :, 1] = op_x  # Green channel
            # rgb_image[:, :, 2] = op_y  # Blue channel

            # augmentation -1
            rgb_image[:, :, 0] = op_x  # Red channel
            rgb_image[:, :, 1] = frame  # Green channel
            rgb_image[:, :, 2] = op_y  # Blue channel

            # augmentation -2
            rgb_image[:, :, 0] = op_x  # Red channel
            rgb_image[:, :, 1] = op_y  # Green channel
            rgb_image[:, :, 2] = frame  # Blue channel

            # augmentation -3
            rgb_image[:, :, 0] = frame  # Red channel
            rgb_image[:, :, 1] = op_y  # Green channel
            rgb_image[:, :, 2] = op_x  # Blue channel

            if os.path.exists(
                    os.path.join(fuse_path, os.listdir(fuse_path)[class_dix], os.listdir(data)[patient_idx] + "_3")) is False:
                os.makedirs(os.path.join(fuse_path, os.listdir(fuse_path)[class_dix], os.listdir(data)[patient_idx]+ "_3"))
            cv2.imwrite(os.path.join(fuse_path, os.listdir(fuse_path)[class_dix], os.listdir(data)[patient_idx] + "_3"
                                     , os.listdir(frames)[img_idx]), rgb_image)

            print("OK")
