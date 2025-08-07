# 统一导入工具包
import os
import warnings
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import cv2
import math
import pyclipper
import imgaug
import imgaug.augmenters as iaa

 
from PIL import Image
from shapely.geometry import Polygon
from collections import OrderedDict
from tqdm import tqdm
from torchvision import transforms
 
import matplotlib
import matplotlib.pyplot as plt
# 是否使用 GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
DEBUG = False  


# 参数设置
class DetOptions():
    def __init__(self):
        self.lr = 0.004
        self.max_epoch = 200
        self.batch_size = 8
        self.num_workers = 8
        self.print_interval = 100
        self.save_interval = 10
        self.train_dir = 'datasets/data/train_imgs'
        self.train_gt_dir = 'datasets/data/train_gts'                 
        self.test_dir = 'datasets/data/test_imgs'
        self.save_dir = 'temp/det_models/'                            # 保存检测模型
        self.saved_model_path = './detector_model'    # 保存最终检测模型
        self.det_res_dir = 'temp/det_res/'                            # 保存测试集检测结
        self.thresh = 0.3                                             # 分割后处理阈值
        self.box_thresh = 0.5                                         # 检测框阈值
        self.max_candidates = 1                                      # 候选检测框数量（本数据集每张图像只有一个文本，因此可置为1）
        self.test_img_short_side = 640                                # 测试图像最短边长度

class RecOptions():
    def __init__(self):
        self.height = 32              # 图像尺寸
        self.width = 100         
        self.voc_size = 21            # 字符数量 '0123456789ABCDEFGHIJ' + 'PADDING'位
        self.decoder_sdim = 512
        self.max_len = 5              # 文本长度
        self.lr = 1.0
        self.milestones = [40, 60]    # 在第 40 和 60 个 epoch 训练时降低学习率
        self.max_epoch = 80
        self.batch_size = 64
        self.num_workers = 8
        self.print_interval = 25
        self.save_interval = 125
        self.train_dir = 'temp/rec_datasets/train_imgs'
        self.test_dir = 'temp/rec_datasets/test_imgs'
        self.save_dir = 'temp/rec_models/'
        self.saved_model_path = './identifer_model'
        self.rec_res_dir = 'temp/rec_res/'
 
 
    def set_(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
 
 
rec_args = RecOptions()
det_args = DetOptions()
    
if DEBUG:
    rec_args.max_epoch = 1
    rec_args.print_interval = 20
    rec_args.save_interval = 1
 
 
    rec_args.batch_size = 10
    rec_args.num_workers = 0

    det_args.max_epoch = 1
    det_args.print_interval = 1
    det_args.save_interval = 1
    det_args.batch_size = 2
    det_args.num_workers = 0

 
 