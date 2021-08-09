from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models import vgg19
import torchvision.transforms as transforms

from glob import glob
import os.path as osp

#定数
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

#学習用のデータセット読み込みクラス
class ImageDataset(Dataset):
    """
    学習のためのDatasetクラス
    32×32の低解像度の本物画像と、
    128×128の本物画像を出力する
    """
    def __init__(self, dataset_dir, hr_shape):
        hr_height, hr_width = hr_shape
        
        # 低解像度の画像を取得するための処理
        self.lr_transform = transforms.Compose([
            transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        # 高像度の画像を取得するための処理
        self.hr_transform = transforms.Compose([
            transforms.Resize((hr_height, hr_height), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        
        self.files = sorted(glob(osp.join(dataset_dir, '*')))
    
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)
        
        return {'lr': img_lr, 'hr': img_hr}
    
    def __len__(self):
        return len(self.files)

#学習中の画像確認とデモ用のデータセット読み込みクラス
class TestImageDataset(Dataset):
    """
    Generatorによる途中経過の確認のためのDatasetクラス
    lr_transformで入力画像を高さと幅それぞれ1/4の低解像度の画像を生成し、
    hr_transformでオリジナルの画像を高解像度の画像として用いる
    """
    def __init__(self, dataset_dir):
        self.hr_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])
        self.files = sorted(glob(osp.join(dataset_dir, '*')))
    
    def lr_transform(self, img, img_size):
        """
        様々な入力画像のサイズに対応するために、
        入力画像のサイズを1/4にするように処理
        """
        img_width, img_height = img_size
        self.__lr_transform = transforms.Compose([
            transforms.Resize((img_height // 4, 
                               img_width // 4), 
                               Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        img = self.__lr_transform(img)
        return img
            
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_size = img.size
        img_lr = self.lr_transform(img, img_size)
        img_hr = self.hr_transform(img)        
        return {'lr': img_lr, 'hr': img_hr}
    
    def __len__(self):
        return len(self.files)