import os
import os.path as osp

from glob import glob
import shutil
from tqdm import tqdm
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def crop_cat():
    # 入力となるデータセットを保存するディレクトリ
    input_dir = 'input'
    # 画像を保存するディレクトリ
    image_dir = osp.join(input_dir, 'images')
    # アノテーションデータを保存するディレクトリ（犬と猫）
    annotations_dir = osp.join(input_dir, 'annotations')
    # アノテーションのリストファイルのパス（犬と猫）
    list_path = osp.join(annotations_dir, 'list.txt')
    
    # データセットのラベル名
    cols = ['file_name', 'class_id', 'species', 'breed_id']
    
    list_path = osp.join("input", "annotations", "list.txt")
    #chapter3/input/annotations/list.txt
    
    labels = []
    with open(list_path, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            if line.startswith('#'):
                continue
            labels.append(line.split(' '))
    f.close()
    labels_df = pd.DataFrame(labels, columns=cols)
    labels_df = labels_df[['file_name', 'species']]
    
    #データを猫に絞る
    cat_label_df = labels_df[labels_df.species=='1']
    cat_label_df = cat_label_df.reset_index(drop=True)
    print(cat_label_df.head(), cat_label_df.shape)
    
    # データセットの学習データを保存するディレクトリ
    train_dir = osp.join("input", "cat_face", 'train')
    # データセットのテストデータを保存するディレクトリ
    test_dir = osp.join("input", "cat_face", 'test')
    # デモ用のデータを保存するディレクトリ
    demo_dir = osp.join("input", "cat_face", 'demo')
    
    #ランダムに切り出す際の定数
    seed = 19930124
    random_crop_times = 4
    # クロップする画像のサイズ
    crop_size = (128, 128) # (height, width)
    dataset_name = 'cat_face'
    
    #ランダムに切り出す関数の定義
    def random_crop(image, crop_size):
        """画像を指定されたサイズになるようにランダムにクロップを行う
    
        Args:
            image (np.array): ランダムクロップする画像
            crop_size (tuple): ランダムクロップするサイズ
    
        Returns:
            np.array: ランダムクロップされた画像
        """
        
        h, w, _ = image.shape
    
        top = np.random.randint(0, h - crop_size[0])
        left = np.random.randint(0, w - crop_size[1])
    
        bottom = top + crop_size[0]
        right = left + crop_size[1]
    
        image = image[top:bottom, left:right, :]
        return image
    
    #シード固定
    np.random.seed(seed)
    
    #画像の分割
    train_df, test_df = train_test_split(cat_label_df, test_size=5, random_state=seed)
    train_df, demo_df = train_test_split(train_df, test_size=1, random_state=seed)
    
    # 学習に用いる画像
    for item in tqdm(train_df.file_name, total=len(train_df)):
        image_name = '{}.jpg'.format(item)
        image_path = osp.join("input", "images", image_name)
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        # 画像のサイズがクロップする画像のサイズより小さいときは処理対象外とする
        if (h < crop_size[0]) | (w < crop_size[1]):
            print('{} size is invalid. h: {},  w: {}'.format(image_name, h, w))
            continue
        for num in range(random_crop_times):
            cropped_image = random_crop(image, crop_size=crop_size)
            image_save_name = '{}_{:03}.jpg'.format(item, num)
            cropped_image_save_path = osp.join(train_dir, image_save_name)
            os.makedirs(osp.dirname(cropped_image_save_path), exist_ok=True)
            cv2.imwrite(cropped_image_save_path, cropped_image)
    # 学習の進度確認の画像
    for item in test_df.file_name:
        image_name = '{}.jpg'.format(item)
        image_path = osp.join("input", "images", image_name)
        
        image_save_path = osp.join(test_dir, image_name)
        os.makedirs(osp.dirname(image_save_path), exist_ok=True)
        shutil.copy(image_path, image_save_path)
    
    # 学習後に超解像を試す画像
    for item in demo_df.file_name:
        image_name = '{}.jpg'.format(item)
        image_path = osp.join("input", "images", image_name)
        
        image_save_path = osp.join(demo_dir, image_name)
        os.makedirs(osp.dirname(image_save_path), exist_ok=True)
        shutil.copy(image_path, image_save_path)