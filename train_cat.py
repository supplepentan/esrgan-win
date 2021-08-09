from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models import vgg19
import torchvision.transforms as transforms

import sys
import json
import random
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

import datasets.imgae_cat
import datasets.datasets_read
import gan

import numpy as np
import os
import os.path as osp

if __name__ == '__main__':
    #datasets.imgae_cat.crop_cat()
    #シード固定
    seed = 19930124
    random.seed(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #関数
    def denormalize(tensors):
      """
      高解像度の生成画像の非正規化を行う
      """
      for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
      return torch.clamp(tensors, 0, 255)
    #パスの定義
    output_dir = "output"
    
    image_train_save_dir = osp.join(output_dir, 'image', 'train')
    image_test_save_dir = osp.join(output_dir, 'image', 'test')
    weight_save_dir = osp.join(output_dir, 'weight')
    param_save_path = osp.join(output_dir, 'param.json')
    
    save_dirs = [image_train_save_dir, image_test_save_dir, weight_save_dir]
    for save_dir in save_dirs:
        print(save_dir)
        os.makedirs(save_dir, exist_ok=True)
    
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    
    
    dataset_dir = "cat_face"
    train_data_dir = osp.join(dataset_dir, 'train')
    test_data_dir = osp.join(dataset_dir, 'test')
    demo_data_dir = osp.join(dataset_dir, 'demo')
    
    #パラメータ
    def save_json(file, save_path, mode):
        """Jsonファイルを保存
        """
        with open(save_path, mode) as outfile:
            json.dump(file, outfile, indent=4)
    class Opts():
        def __init__(self):
            self.n_epoch = 50
            self.residual_blocks = 23
            self.lr = 0.0002
            self.b1 = 0.9
            self.b2 = 0.999
            self.batch_size = 8
            self.n_cpu = 8
            self.warmup_batches = 500
            self.lambda_adv = 5e-3
            self.lambda_pixel = 1e-2
            self.pretrained = False
            self.dataset_name = 'cat'
            self.sample_interval = 100
            self.checkpoint_interval = 1000
            self.hr_height = 128
            self.hr_width = 128
            self.channels = 3
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        def to_dict(self):
            parameters = {
                'n_epoch': self.n_epoch,
                'hr_height': self.hr_height,
                'residual_blocks': self.residual_blocks,
                'lr': self.lr,
                'b1': self.b1,
                'b2': self.b2,
                'batch_size': self.batch_size,
                'n_cpu': self.n_cpu,
                'warmup_batches': self.warmup_batches,
                'lambda_adv': self.lambda_adv,
                'lambda_pixel': self.lambda_pixel,
                'pretrained': self.pretrained,
                'dataset_name': self.dataset_name,
                'sample_interval': self.sample_interval,
                'checkpoint_interval': self.checkpoint_interval,
                'hr_height': self.hr_height,
                'hr_width': self.hr_width,
                'channels': self.channels,
                'device': str(self.device),
            }
            return parameters
    opt = Opts()
    save_json(opt.to_dict(), param_save_path, 'w')
    
    hr_shape = (opt.hr_height, opt.hr_height)
    
    train_data_dir = osp.join("input", "cat_face", "train")
    
    #データセット
    train_dataloader = DataLoader(
        datasets.datasets_read.ImageDataset(train_data_dir, hr_shape=hr_shape),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        #num_workers = 0
    )
    
    test_dataloader = DataLoader(
        datasets.datasets_read.TestImageDataset(test_data_dir),
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_cpu,
        #num_workers = 0
    )
    
    # ESRGANを呼び出す
    esrgan = gan.ESRGAN(opt, hr_shape, log_dir)
    
    
    for epoch in range(1, opt.n_epoch + 1):
        for batch_num, imgs in enumerate(train_dataloader):
            batches_done = (epoch - 1) * len(train_dataloader) + batch_num
            # 事前学習
            if batches_done <= opt.warmup_batches:
                esrgan.pre_train(imgs, batches_done, batch_num, epoch, opt)
            # 本学習
            else:
                esrgan.train(imgs, batches_done, batch_num, opt, epoch)
            # 高解像度の生成画像の保存
            if batches_done % opt.sample_interval == 0:
                for i, imgs in enumerate(test_dataloader):
                    esrgan.save_image(imgs, batches_done)
            # 学習した重みの保存
            if batches_done % opt.checkpoint_interval == 0:
                esrgan.save_weight(batches_done, weight_save_dir)