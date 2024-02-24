import sys
sys.path.append('/home/gzq/Code/scribbles/my_github_code')
import os.path as osp
import cv2
import os
import numpy as np
import torch
from torch.utils import data
import model.transforms as T
from scipy.ndimage import zoom, rotate

class DriveOriSeg(data.Dataset):
    def __init__(self, img_folder, lab_folder, train=True):
        self._transforms = T.Compose([
            T.ToTensor(),
            T.Normalize()])
        img_paths = sorted(os.listdir(img_folder))
        lab_paths = sorted(os.listdir(lab_folder))
        self.data_list = []
        self.train = train
        for i in range(0, len(img_paths)):
            img_path = img_folder + img_paths[i]
            lab_path = lab_folder + lab_paths[i]
            img = cv2.imread(str(img_path), 0)
            img_name = img_paths[i][:-4]
            lab_name = lab_paths[i][:-4]
            img = (img - img.min()) / (img.max() - img.min())
            lab = cv2.imread(str(lab_path), 0)
            if (lab == 128).sum() != 0:
                lab[lab == 0] = 2
                lab[lab == 128] = 0
                lab[lab == 255] = 1
            else:
                lab[lab == 255] = 1
            print(img_name, lab_name)
            self.data_list.append((img, lab, img_name))
    
    def __getitem__(self, idx):
        img, lab, img_name = self.data_list[idx]
        if self.train:
            if np.random.rand() < 0.75:
                rot = np.random.rand() * 360
                img = rotate(img, rot, order=1, reshape=False, mode='constant',cval=0)
                lab = rotate(lab, rot, order=0, reshape=False, mode='constant',cval=0)
        mean = img.mean()
        std = img.std()
        img = (img - mean) / std
        img = np.expand_dims(img, 0)
        lab = np.expand_dims(lab, 0)
        return {'img':img, 'lab':lab, 'img_name':img_name}

    def __len__(self):
        return len(self.data_list)

class Stage2Set(data.Dataset):
    def __init__(self, img_folder, lab_folder, stage1_folder, train=True):
        self._transforms = T.Compose([
            T.ToTensor(),
            T.Normalize()])
        img_paths = sorted(os.listdir(img_folder))
        lab_paths = sorted(os.listdir(lab_folder))
        stage1_paths = sorted(os.listdir(stage1_folder))
        self.data_list = []
        self.train = train
        for i in range(0, len(img_paths)):
            img_path = img_folder + img_paths[i]
            lab_path = lab_folder + lab_paths[i]
            stage1_path = stage1_folder + stage1_paths[i]
            if not train:
                stage1 = np.zeros((3, 512, 512))
            else:
                stage1 = np.load(stage1_path)
            img = cv2.imread(str(img_path), 0)
            img_name = img_paths[i][:-4]
            lab_name = lab_paths[i][:-4]
            img = (img - img.min()) / (img.max() - img.min())
            lab = cv2.imread(str(lab_path), 0)
            if (lab == 128).sum() != 0:
                lab[lab == 0] = 2
                lab[lab == 128] = 0
                lab[lab == 255] = 1
            else:
                lab[lab == 255] = 1
            print(img_name, lab_name)
            self.data_list.append((img, lab, stage1, img_name))
    
    def __getitem__(self, idx):
        img, lab, stage1, img_name = self.data_list[idx]
        stage1_sup, stage1_ori_mask = stage1[:2], stage1[2]
        if self.train:
            if np.random.rand() < 0.75:
                rot = np.random.rand() * 360
                img = rotate(img, rot, order=1, reshape=False, mode='constant',cval=0)
                lab = rotate(lab, rot, order=0, reshape=False, mode='constant',cval=0)
        mean = img.mean()
        std = img.std()
        img = (img - mean) / std
        img = np.expand_dims(img, 0)
        lab = np.expand_dims(lab, 0)
        stage1_ori_mask = np.expand_dims(stage1_ori_mask, 0)
        return {'img':img, 'lab':lab, 'img_name':img_name, 'stage1_sup':stage1_sup, 'stage1_mask':stage1_ori_mask}

    def __len__(self):
        return len(self.data_list)
