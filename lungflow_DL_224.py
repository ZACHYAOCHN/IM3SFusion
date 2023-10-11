#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zach Yao
@license: None
@contact: yaoliangchn@qq.com
@software: Pycharm Community.
@file: lungflow_DL_224.py
@time: 2023/5/4 22:27
@desc: US image (1D) + Optical Flow (2D).
'''
import os
from typing import Dict, Any
# from pathlib import Path
import albumentations
import numpy as np

import cv2
import torch
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from pytorch_lightning import LightningDataModule

# 将此函数删掉，并整合到其他函数中。
class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'lungVideo':
            # root_dir = 'F:\\Dataset\\1.Medical_dataset\\lungBlas_301\\patient_split'
            root_dir = 'H:\\Dataset\\1.Medical_dataset\\lungBlas_301\\shape_split'
            # output_dir = './path/to/VAR/patient_crop'
            output_dir = "H:\\Dataset\\1.Medical_dataset\\lungBlas_301\\xyzchannel_merge_split\\split_11"
            # output_dir = "F:\\Dataset\\1.Medical_dataset\\lungBlas_301\\xyzchannel_merge_split\\split_12"
            return root_dir, output_dir

        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return "./ucf101-caffe.pth"



class lung_ds(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            ds_name (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, ds_name='lungVideo', split='train', clip_len=16, preprocess=False):
        self.root_dir, self.output_dir = Path.db_dir(ds_name)   # 获取数据集的源路径和输出路径
        folder = os.path.join(self.output_dir, split)    # 获取对应分组的的路径，即train，test，val的路径
        self.clip_len = clip_len                            # 一个片段多少帧
        # self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 256  # 128
        self.resize_width = 256  # 171
        self.crop_size = 224 # 112

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        # 把如下代码注释后，直接放生成的光流融合视频图像的地址在：output_dir = 'XXX',需要对数据集进行随机划分。
        # if (not self.check_preprocess()) or preprocess:
        #     print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(ds_name))
        #     self.preprocess()

        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if os.path.exists('dataloader/lungblast_label.txt'):
            with open('dataloader/lungblast_label.txt', 'w') as f:
                for id, label in enumerate(sorted(self.label2index)):
                    f.writelines(str(id+1) + ' ' + label + '\n')

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    def check_preprocess(self):
        # TODO: Check image size in output_dir
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                            sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)  #   读取图片
                if np.shape(image)[0] != 256 or np.shape(image)[1] != 256:
                    return False
                else:
                    break

            if ii == 2:
                break

        return True

    '''
    def preprocess(self):# 预处理视频
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            os.makedirs(os.path.join(self.output_dir, 'train'))
            os.makedirs(os.path.join(self.output_dir, 'val'))
            os.makedirs(os.path.join(self.output_dir, 'test'))

        # Split train/val/test sets
        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)    # 得到分好的数据集路径 train, val, test
            for label in os.listdir(file_path):
                label_path = os.path.join(file_path, label)
                video_files = [name for name in os.listdir(label_path)]  # 得到每个视频的名字，类型为list，中括号不能省
                outfile_dir = os.path.join(self.output_dir, file, label)
                if not os.path.exists(outfile_dir):
                    os.mkdir(outfile_dir)
                for video in video_files:
                    self.process_video(video, os.path.join(file,label), outfile_dir)

        print('Preprocessing finished.')
    '''

    '''
    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        # video_filename = video.split('.')[0] + video.split('.')[1] + video.split('.')[2]   # 由于video的名称为：A (1).mp42.0.avi，由此filename取为A (1)mp420
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video)) # 读取视频

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # 读取视频有多少帧

        # print(os.path.join(self.root_dir, action_name, video)+'frame_count:', frame_count)

        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # 读取视频宽度
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames,确定隔几帧取一张，取够16帧
        EXTRACT_FREQUENCY = 4   # 隔EXTRACT_FREQUENCY帧取一次数据，默认为4，取不够再减小
        if frame_count // EXTRACT_FREQUENCY < 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY < 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY < 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue
            # 读取视频的每一帧
            if count % EXTRACT_FREQUENCY == 0:    # 判断这一帧是不是隔EXTRACT_FREQUENCY一取
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    # LL-START
                    # resize_height_tem = 300
                    # resize_width_tem = 400
                    # height_center = frame_height//2
                    # height_start = height_center - (resize_height_tem//2) + 1
                    # height_end = height_center + (resize_height_tem-resize_height_tem//2)
                    # width_center = frame_width//2
                    # width_start = width_center - (resize_width_tem//2) + 1
                    # width_end = width_center + (resize_width_tem-resize_width_tem//2)
                    # frame = frame[height_start:height_end, width_start:width_end]
                    # LL-END
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()
    '''

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):
        # compute the mean and std for each buffer

        # mean = torch.zeros(3)
        # std = torch.zeros(3)
        # for i, frame in enumerate(buffer):
        #     for d in range(3):
        #         mean[d] += frame[:, :, d].mean()
        #         std[d] += frame[:, :, d].std()
        # # print('meann:', mean)
        # mean.div_(self.clip_len)
        # # print('mean:', mean)
        # std.div_(self.clip_len)
        # normalize
        # global mean and std
        # mean = [48.7824, 51.4731, 49.7559]
        # std = [37.8534, 38.2911, 38.0883]
        # global mean and std when no crop
        mean_nocrop = [29.4091, 31.1655, 30.0138]
        std_nocrop = [39.5987, 40.1935, 39.7596]
        # transforms_norm = transforms.Normalize(mean, std)
        # for i, frame in enumerate(buffer):
        #     # [H W C] TO [C H W]
        #     frame = transforms.ToTensor()(frame)
        #     frame = transforms_norm(frame)
        #     frame = np.array(frame)
        #     # [C H W] TO [H W C]
        #     frame = frame.transpose((1, 2, 0))
        #     # print('frame:',frame)
        #     buffer[i] = frame
        # minus mean
        for i, frame in enumerate(buffer):
            # need to be  modif
            # print('the size of frame:',frame.shape)
            # frame -= np.array([[[90.0, 98.0, 102.0]]])
            frame -= np.array([[[mean_nocrop[0], mean_nocrop[1], mean_nocrop[2]]]])
            # frame -= np.array([[[48.5383, 51.2226, 49.5110]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            ## 此处读进去 frame
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            # frame = np.array(cv2.imread(frame_name, cv2.IMREAD_GRAYSCALE)).astype(np.float64)
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        # time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        # buffer = buffer[0:clip_len, :, :, :]
        buffer = buffer[0:clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer


class build_lung_dl(LightningDataModule):
    def __init__(self,
                 # root: str = "F:\\Dataset\\1.Medical_dataset\\Dataset_BUSI\\Random_split_1\\",
                 batch_size: int = 64,
                 num_workers: int = 32,
                 seed: int = 123,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def _get_ds(self):
        """
            准备Dataset,
        """
        ds_train = lung_ds(ds_name='lungVideo', split='train', clip_len=16)
        ds_vali = lung_ds(ds_name='lungVideo', split='val', clip_len=16)
        ds_test = lung_ds(ds_name='lungVideo', split='test', clip_len=16)

        return ds_train, ds_vali, ds_test

    def lungVideo_dl(self):
        ds_train, ds_vali, ds_test = self._get_ds()

        dl_train = DataLoader(dataset=ds_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)  # True
        dl_test = DataLoader(dataset=ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        dl_vali = DataLoader(dataset=ds_vali, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return dl_train, dl_vali, dl_test
