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

import sys
import cv2
import argparse
import os
import sys
import subprocess
import glob

from options.train_options import Opts
from datasets.imgae_cat import demo_crop_cat
from datasets.imgae_cat import crop_cat


# シード固定
seed = 19930124
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# パスの定義
# 入力となるデータセットを保存するディレクトリ
input_dir = "input"

output_dir = "output"
image_train_save_dir = os.path.join(output_dir, 'image', 'train')
image_test_save_dir = os.path.join(output_dir, 'image', 'test')
weight_save_dir = os.path.join(output_dir, 'weight')
param_save_path = os.path.join(output_dir, 'param.json')

log_dir = './logs'
#os.makedirs(log_dir, exist_ok=True)

dataset_dir = "cat_face"
train_data_dir = os.path.join(dataset_dir, 'train')
test_data_dir = os.path.join(dataset_dir, 'test')
demo_data_dir = os.path.join(dataset_dir, 'demo')

# パラメータ


def save_json(file, save_path, mode):  # Jsonファイルを保存
    with open(save_path, mode) as outfile:
        json.dump(file, outfile, indent=4)

# 関数


def denormalize(tensors):  # 高解像度の生成画像の非正規化を行う
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)

def pre():
    print(input_dir)
    crop_cat(input_dir)

def main(args):
    save_dirs = [image_train_save_dir, image_test_save_dir, weight_save_dir]
    for save_dir in save_dirs:
        print(save_dir)
        os.makedirs(save_dir, exist_ok=True)

    opt = Opts(args)
    save_json(opt.to_dict(), param_save_path, 'w')

    hr_shape = (opt.hr_height, opt.hr_height)

    train_data_dir = os.path.join(input_dir, "cat_face", "train")

    # データセット
    train_dataloader = DataLoader(
        datasets.datasets_read.ImageDataset(train_data_dir, hr_shape=hr_shape),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu
    )

    test_dataloader = DataLoader(
        datasets.datasets_read.TestImageDataset(test_data_dir),
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_cpu
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ESRGAN Training Mode CAT")
    parser.add_argument("--n_epoch", action="store", default=50, help="")
    parser.add_argument("--residual_blocks",
                        action="store", default=23, help="")
    parser.add_argument("--lr", action="store", default=0.002, help="")
    parser.add_argument("--b1", action="store", default=0.9, help="")
    parser.add_argument("--b2", action="store", default=0.999, help="")
    parser.add_argument("--batch_size", action="store", default=8, help="")
    parser.add_argument("--n_cpu", action="store", default=8, help="")
    parser.add_argument("--warmup_batches", action="store",
                        default=500, help="")
    parser.add_argument("--lambda_adv", action="store", default=5e-3, help="")
    parser.add_argument("--lambda_pixel", action="store",
                        default=1e-2, help="")
    parser.add_argument("--pretrained", action="store", default=False, help="")
    parser.add_argument("--dataset_name", action="store",
                        default='cat', help="")
    parser.add_argument("--sample_interval",
                        action="store", default=100, help="")
    parser.add_argument("--checkpoint_interval",
                        action="store", default=1000, help="")
    parser.add_argument("--hr_height", action="store", default=128, help="")
    parser.add_argument("--hr_width", action="store", default=128, help="")
    parser.add_argument("--channels", action="store", default=3, help="")
    parser.add_argument("--device", action="store", default=torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'), help="")
    args = parser.parse_args()
    pre()
    main(args)
