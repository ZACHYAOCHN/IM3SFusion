#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zach Yao
@license: None
@contact: yaoliangchn@qq.com
@software: Pycharm Community.
@file: VideoSwinT.py
@time: 2023/3/31 17:14
@desc: Using Video Swin Transformer for classification
        !!! 目前状态 ！！！
        刚从video_cls_ori.py复制过来，方法还没改。
'''

import pytorch_lightning as pl
import torch
from torch import nn
import numpy as np
import torch.optim as optim

## Import models
import pytorchvideo.models.resnet
from mmaction.models.backbones.swin_transformer import SwinTransformer3D
# from mmaction.models.backbones.swin_transformer_ori import SwinTransformer3D

from torchmetrics.classification import Accuracy as ACC
from torchmetrics.classification import Precision as Pre
from torchmetrics.classification import Recall as Rec
from torchmetrics.classification import ConfusionMatrix, F1Score
from torchmetrics.utilities.data import to_onehot

acc_cls = ACC(task='binary', num_classes=2).to(device='cuda')
Precision_cls = Pre(task='binary', num_classes=2).to(device='cuda')
recall_cls = Rec(task='binary', num_classes=2).to(device='cuda')
confusion_matrix_cls = ConfusionMatrix(task='binary', num_classes=2).to(device='cuda')
F1score_cls = F1Score(task='binary', num_classes=2).to(device='cuda')

from models.VideoCls.models_ori.C3D_model import C3D


class MViT(pl.LightningModule):
    """
    Multiscale Vision Transformers
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik,
    Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """
    def __init__(self,
                 num_classes: int = 2,
                 # models: str = 'swinT',
                 # head: str = 'seghead',
                 ):
        super().__init__()
        self.num_classes = num_classes
        # self.model = pytorchvideo.models.resnet.create_resnet(
        #         input_channel=3,
        #         model_num_class=self.num_classes,
        #     )
        self.model = pytorchvideo.models.vision_transformers.create_multiscale_vision_transformers(
            input_channels=3,
            head_num_classes=2,
            spatial_size=112,
            temporal_size=16
        )

        # self.example_input_array = torch.zeros(1, 3, 224, 224)

    def cls_loss(self, cls_logits, label):
        return nn.CrossEntropyLoss(weight=[0.31,1])(cls_logits, label)

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        # getting outputs
        data = batch[0].to(device='cuda', dtype=torch.float32)
        label = batch[1].to(device='cuda', dtype=torch.int64)

        output = self.forward(data)
        loss = self.cls_loss(output, label)

        label1 = to_onehot(label, num_classes=2)
        cls_ACC = acc_cls(output, label1)

        # logging
        self.log('tr_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('tr_ACC', cls_ACC, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        data = batch[0].to(device='cuda', dtype=torch.float32)
        label = batch[1].to(device='cuda', dtype=torch.int64)

        output = self.forward(data)
        loss = self.cls_loss(output, label)
        label1 = to_onehot(label, num_classes=2)
        cls_ACC = acc_cls(output, label1)

        # logging
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_ACC', cls_ACC, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_test_epoch_start(self):
        self.test_pred = None
        return None

    def test_step(self, batch, batch_idx):
        # getting outputs
        data = batch[0].to(device='cuda', dtype=torch.float32)
        label = batch[1].to(device='cuda', dtype=torch.int64)

        output = self.forward(data)
        loss = self.cls_loss(output, label)

        ## Evaluation Metrics for Classification
        label1 = to_onehot(label, num_classes=2)
        cls_ACC = acc_cls(output, label1)
        # cls_Precision = Precision_cls(output, label)
        # cls_Recall = recall_cls(output, label)
        # cls_confusionM = confusion_matrix_cls(output, label)
        # cls_F1 = F1score_cls(output, label)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_ACC', cls_ACC, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('test_Precision_cls', cls_Precision, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('test_Recall_cls', cls_Recall, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('test_F1score_cls', cls_F1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=5e-4) # weight_decay=1e-6时IoU掉一个点
    #     self.lr_scheduler = CosineWarmupScheduler(
    #         optimizer, warmup=20, max_iters=100)  # 之前是100，200 换成 20， 100  ## warmup: warmup起作用的最大epoch，max_iters: 最大迭代epoch.
    #     return optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=5e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                                            gamma=0.1)
        # self.lr_scheduler = CosineWarmupScheduler(optimizer, warmup=10, max_iters=100)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration


class VideoSwinT(pl.LightningModule):
    """
    Video Swin Transformer
    """
    def __init__(self,
                 num_classes: int = 2,
                 # models: str = 'swinT',
                 # head: str = 'seghead',
                 ):
        super().__init__()
        self.num_classes = num_classes
        # self.model = pytorchvideo.models.resnet.create_resnet(
        #         input_channel=3,
        #         model_num_class=self.num_classes,
        #     )
        self.model = SwinTransformer3D(pretrained="pretrained_weights\\swin_tiny_patch244_window877_kinetics400_1k.pth",
                                       pretrained2d=False, patch_norm=True, use_checkpoint=False)
        # self.model = SwinTransformer3D(pretrained=None,
        #                                pretrained2d=False, patch_norm=True, use_checkpoint=False)
        ## 错误的地址，用于测试是否读入weights.
        # self.model = SwinTransformer3D(pretrained="pretrained_weights\\tiny_patch244_window877_kinetics400_1k.pth",
        #                                pretrained2d=False, patch_norm=True, use_checkpoint=True)

        self.example_input_array = torch.zeros(1, 3, 16, 224, 224)

    def cls_loss(self, cls_logits, label):
        # return nn.CrossEntropyLoss(weight=torch.tensor([1, 0.31]).to('cuda'), label_smoothing=0)(cls_logits, label)
        # return nn.CrossEntropyLoss(weight=torch.tensor([1., 3.5]).to('cuda'), label_smoothing=0.2)(cls_logits, label)
        return nn.CrossEntropyLoss(weight=torch.tensor([1., 1.]).to('cuda'), label_smoothing=0.1)(cls_logits, label)


    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        # getting outputs
        data = batch[0].to(device='cuda', dtype=torch.float32)
        label = batch[1].to(device='cuda', dtype=torch.int64)

        output = self.forward(data)
        loss = self.cls_loss(output, label)

        label1 = to_onehot(label, num_classes=2)
        cls_ACC = acc_cls(output, label1)

        # logging
        self.log('tr_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('tr_ACC', cls_ACC, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        data = batch[0].to(device='cuda', dtype=torch.float32)
        label = batch[1].to(device='cuda', dtype=torch.int64)

        output = self.forward(data)
        loss = self.cls_loss(output, label)
        label1 = to_onehot(label, num_classes=2)
        cls_ACC = acc_cls(output, label1)

        # logging
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_ACC', cls_ACC, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_test_epoch_start(self):
        self.test_pred = None
        return None

    def test_step(self, batch, batch_idx):
        # getting outputs
        data = batch[0].to(device='cuda', dtype=torch.float32)
        label = batch[1].to(device='cuda', dtype=torch.int64)

        output = self.forward(data)
        loss = self.cls_loss(output, label)

        ## Evaluation Metrics for Classification
        label1 = to_onehot(label, num_classes=2)
        cls_ACC = acc_cls(output, label1)
        # cls_Precision = Precision_cls(output, label)
        # cls_Recall = recall_cls(output, label)
        # cls_confusionM = confusion_matrix_cls(output, label)
        # cls_F1 = F1score_cls(output, label)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_ACC', cls_ACC, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('test_Precision_cls', cls_Precision, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('test_Recall_cls', cls_Recall, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('test_F1score_cls', cls_F1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=5e-4) # weight_decay=1e-6时IoU掉一个点, lr=5e-6
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=10, max_iters=100)  # 之前是100，200 换成 20， 100  ## warmup: warmup起作用的最大epoch，max_iters: 最大迭代epoch.
        return optimizer

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=5e-4)
    #     self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,
    #                                                         gamma=0.1)
    #     # self.lr_scheduler = CosineWarmupScheduler(optimizer, warmup=10, max_iters=100)
    #     return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration


## 根据教程编写（https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html）
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor