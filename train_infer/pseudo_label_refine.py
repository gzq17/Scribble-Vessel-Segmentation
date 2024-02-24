import numpy as np
import sys
sys.path.append('/home/guozhanqiang/med27_code/scribbles/my_github_code')
import torch
import cv2
import matplotlib.pyplot as plt
import copy
from skimage import transform
import skimage
from skimage.morphology import skeletonize
import math
from tqdm import tqdm
import os
from torch import nn
from torch.autograd import Variable
import scipy.ndimage as ndi
import argparse
from model.parser import get_args_parser
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser('REIVE', parents=[get_args_parser()])
args = parser.parse_args()
root_path = args.root_path

def calc_orientation_graident(img, win_size=16, stride=8):
    Gx, Gy = np.gradient(img.astype(np.float32))
    Gxx = ndi.gaussian_filter(Gx ** 2, win_size / 4)
    Gyy = ndi.gaussian_filter(Gy ** 2, win_size / 4)
    Gxy = ndi.gaussian_filter(-Gx * Gy, win_size / 4)
    coh = np.sqrt((Gxx - Gyy) ** 2 + 4 * Gxy ** 2) 
    if stride != 1:
        Gxx = ndi.uniform_filter(Gxx, stride)[::stride, ::stride]
        Gyy = ndi.uniform_filter(Gyy, stride)[::stride, ::stride]
        Gxy = ndi.uniform_filter(Gxy, stride)[::stride, ::stride]
        coh = ndi.uniform_filter(coh, stride)[::stride, ::stride]
    ori = np.arctan2(2 * Gxy, Gxx - Gyy) * 90 / np.pi
    return ori, coh

def Gedge_map(im):
    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    conv_op.weight.data = torch.from_numpy(sobel_kernel).cuda()
    edge_detect = torch.abs(conv_op(Variable(im)))

    conv_op1 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')
    sobel_kernel1 = sobel_kernel1.reshape((1, 1, 3, 3))
    conv_op1.weight.data = torch.from_numpy(sobel_kernel1).cuda()
    edge_detect1 = torch.abs(conv_op1(Variable(im)))

    conv_op2 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel2 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype='float32')
    sobel_kernel2 = sobel_kernel2.reshape((1, 1, 3, 3))
    conv_op2.weight.data = torch.from_numpy(sobel_kernel2).cuda()
    edge_detect2 = torch.abs(conv_op2(Variable(im)))

    conv_op3 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel3 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype='float32')
    sobel_kernel3 = sobel_kernel3.reshape((1, 1, 3, 3))
    conv_op3.weight.data = torch.from_numpy(sobel_kernel3).cuda()
    edge_detect3 = torch.abs(conv_op3(Variable(im)))
    # print(conv_op.weight.size())
    # print(conv_op, '\n')

    sobel_out = edge_detect+edge_detect1+edge_detect2+edge_detect3

    return sobel_out

def get_points_v(min_l, max_l, edge_patch, h_s, angle):
    points_have = []
    points = np.array([[0, 0]])
    points_v = []
    for l in np.arange(min_l, max_l, 0.5):
        x_new = round(h_s + l * math.cos(math.radians(angle)))
        y_new = round(h_s + l * math.sin(math.radians(angle)))
        ss = str(x_new) + '+' + str(y_new)
        if ss in points_have:
            continue
        points_have.append(ss)
        points = np.concatenate([points, np.array([[x_new, y_new]])], axis=0)
        points_v.append(edge_patch[y_new, x_new])
    points = points[1:]
    points_v = np.array(points_v)
    points_v_ave = (points_v[:-2] + points_v[1:-1] + points_v[2:]) / 3
    points = points[1:-1]
    return points_v_ave, points, points_v

def get_points_v2(min_l, max_l, edge_patch, h_s, angle):
    points = np.array([[0, 0]])
    for l in np.arange(min_l, max_l, 0.5):
        x_new = h_s + l * math.cos(math.radians(angle))
        y_new = h_s + l * math.sin(math.radians(angle))
        points = np.concatenate([points, np.array([[x_new, y_new]])], axis=0)
    points = points[1:]
    return points

def find_bound(img, lbl, fore_stage, hard=False):
    sz, max_l, h_s = 512, 9, 16
    img_use = copy.deepcopy(img[np.newaxis, np.newaxis, :, :])
    img_use = (img_use - img_use.min()) / (img_use.max() - img_use.min())
    img_use = torch.from_numpy(img_use).float().cuda()
    edge = Gedge_map(img_use)[0, 0].detach().cpu().numpy()
    edge = (edge - edge.min()) / (edge.max() - edge.min())
    edge[edge > 0.5] = 0.5
    edge = (edge - edge.min()) / (edge.max() - edge.min())
    fore_thin = skeletonize(fore_stage)

    fore_thin = fore_thin.astype(np.uint8)
    [label, num] = skimage.measure.label(fore_thin, return_num=True)
    fore_thin_new = copy.deepcopy(fore_thin)
    for i in range(1, num + 1):
        temp = (label == i).sum()
        if temp < 20:
            fore_thin_new[label == i] = 0
    ori_img = copy.deepcopy(fore_stage)
    ori_img[ori_img == 0] = 10
    ori_img[ori_img == 1] = 255
    ori, coh = calc_orientation_graident(ori_img, win_size=16, stride=8)
    index = np.where(fore_thin_new == 1)
    fore_thin_new_new = copy.deepcopy(fore_thin_new)
    for i in range(0, index[0].shape[0]):
        xx, yy = index[0][i], index[1][i]
        if xx < 5 or xx + 6 > fore_thin_new.shape[0] or yy < 5 or yy + 6 > fore_thin_new.shape[1]:
            continue
        if fore_thin_new[xx-1:xx+2, yy-1:yy+2].sum() > 3:
            fore_thin_new_new[xx-5:xx+6, yy-5:yy+6] = 0
    fore_thin_new = fore_thin_new_new
    index = np.where(fore_thin_new == 1)
    add_stage = np.zeros(lbl.shape) + 0.5
    add_stage_w = np.zeros(lbl.shape) + 1
    for i in range(0, index[0].shape[0]):
        xx, yy = index[0][i], index[1][i]
        x_b, y_b = xx - h_s, yy - h_s
        x_e, y_e = xx + h_s + 1, yy + h_s + 1
        if x_b < 0 or y_b < 0 or x_e > sz or y_e > sz:
            continue
        x_ori, y_ori = round(xx / 8), round(yy / 8)
        edge_patch = copy.deepcopy(edge[x_b:x_e, y_b:y_e])
        lbl_patch = copy.deepcopy(lbl[x_b:x_e, y_b:y_e])
        fore_thin_patch = copy.deepcopy(fore_thin_new[x_b:x_e, y_b:y_e])
        angle = ori[x_ori, y_ori]
        if angle < 0:
            an = angle + 90
        else:
            an = angle - 90
        points_v_0, points_0, points_v_ori_0 = get_points_v(-max_l, 0.5, edge_patch, h_s, an)
        points_v_1, points_1, points_v_ori_1 = get_points_v(0, max_l + 0.5, edge_patch, h_s, an)
        index0 = np.where(points_v_0 == points_v_0.max())[0][0]
        index1 = np.where(points_v_1 == points_v_1.max())[0][0]
        if index0 == 0 or index1 == points_1.shape[0] - 1:
            continue
        if math.fabs(points_0.shape[0] - 1 - index0 - index1) > 1:
            continue
        a, b = 0, 0
        if index0 == points_0.shape[0] - 1:
            if points_v_ori_0[index0] >= points_v_ori_0[index0 + 2]:
                a = 1
        else:
            if points_v_0[index0 - 1] >= points_v_0[index0 + 1]:
                a = 1
        if index1 == 0:
            if points_v_ori_1[0] <= points_v_ori_1[2]:
                b = 1
        else:
            if points_v_1[index1 - 1] <= points_v_1[index1 + 1]:
                b = 1
        new_stage = np.zeros(lbl_patch.shape) + 0.5
        new_stage_w = np.zeros(lbl_patch.shape)
        for ii in range(0, points_0.shape[0]):
            if ii < index0:
                temp = (index0 - ii) * 0.5 / (index0 + 1) + 0.5
                new_stage[points_0[ii, 1], points_0[ii, 0]] = 0
                add_stage[points_0[ii, 1] + x_b, points_0[ii, 0] + y_b] = 0
            elif ii > index0:
                temp = 0.5 + 0.5 / (points_0.shape[0] - index0) * (ii - index0)
                new_stage[points_0[ii, 1], points_0[ii, 0]] = 1
                add_stage[points_0[ii, 1] + x_b, points_0[ii, 0] + y_b] = 1
            else:
                temp = 0.5
                if hard:
                    new_stage[points_0[ii, 1], points_0[ii, 0]] = a
                    add_stage[points_0[ii, 1] + x_b, points_0[ii, 0] + y_b] = a
                else:
                    new_stage[points_0[ii, 1], points_0[ii, 0]] = 0.5
                    add_stage[points_0[ii, 1] + x_b, points_0[ii, 0] + y_b] = 0.5
            add_stage_w[points_0[ii, 1] + x_b, points_0[ii, 0] + y_b] = temp
            new_stage_w[points_0[ii, 1], points_0[ii, 0]] = temp
        for ii in range(0, points_1.shape[0]):
            if ii < index1:
                temp = (index1 - ii) * 0.5 / (index1 + 1) + 0.5
                new_stage[points_1[ii, 1], points_1[ii, 0]] = 1
                add_stage[points_1[ii, 1] + x_b, points_1[ii, 0] + y_b] = 1
            elif ii > index1:
                temp = 0.5 + 0.5 / (points_1.shape[0] - index1) * (ii - index1)
                new_stage[points_1[ii, 1], points_1[ii, 0]] = 0
                add_stage[points_1[ii, 1] + x_b, points_1[ii, 0] + y_b] = 0
            else:
                temp = 0.5
                if hard:
                    new_stage[points_1[ii, 1], points_1[ii, 0]] = b
                    add_stage[points_1[ii, 1] + x_b, points_1[ii, 0] + y_b] = b
                else:
                    new_stage[points_1[ii, 1], points_1[ii, 0]] = 0.5
                    add_stage[points_1[ii, 1] + x_b, points_1[ii, 0] + y_b] = 0.5
            add_stage_w[points_1[ii, 1] + x_b, points_1[ii, 0] + y_b] = temp
            new_stage_w[points_1[ii, 1], points_1[ii, 0]] = temp
    return add_stage, add_stage_w

def main():
    set_list = ['train/', 'val/', 'TestSet/']
    for img_set in set_list:
        parent_path = args.root_path + args.dataset + '/' + img_set
        img_path = parent_path + 'images/'
        label_path = parent_path + 'labels/'
        if img_set == 'train/':
            label_path = parent_path + 'annotations/'
        stage1_path = args.root_path + args.dataset + '_res/' + 'stage1_res/stage1_res/'
        hard = False
        out_path = parent_path + 'stage1_res_soft/'
        out_w_path = parent_path + 'stage1_res_soft_w/'
        os.makedirs(out_path, exist_ok=True)
        os.makedirs(out_w_path, exist_ok=True)
        name_list = sorted(os.listdir(img_path))
        put_list = ['fore ori err', 'fore add err', 'fore new err', 'fore add num', 
                    'back ori err', 'back add err', 'back new err', 'back add num']
        sum_list = [0, 0, 0, 0, 0, 0, 0, 0]
        k = 0
        for name in name_list:
            print('{}/{}'.format(k+1, len(name_list)))
            name = name[:-4]
            img = cv2.imread(img_path + name + '.png', 0)
            lbl = cv2.imread(label_path + name + '_label.png', 0)
            lbl[lbl > 0] = 1
            stage = np.load(stage1_path + name + '.npy')
            new_stage = copy.deepcopy(stage)
            out_name = out_path + name + '.npy'
            out_w_name = out_w_path + name + '_w.npy'
            fore_stage = stage[1]
            add_stage, add_stage_w = find_bound(img, lbl, fore_stage, hard)
            err1 = ((add_stage == 1) & (lbl == 0)).sum() / (add_stage == 1).sum()
            ori_err1 = ((stage[1] == 1) & (lbl == 0)).sum() / (stage[1] == 1).sum()
            err2 = ((add_stage == 0) & (lbl == 1)).sum() / (add_stage == 0).sum()
            ori_err2 = ((stage[0] == 1) & (lbl == 1)).sum() / (stage[0] == 1).sum()
            new_stage[0][add_stage == 0] = 1
            new_stage[0][add_stage == 1] = 0
            new_stage[1][add_stage == 0] = 0
            new_stage[1][add_stage == 1] = 1
            new_err1 = ((new_stage[1] == 1) & (lbl == 0)).sum() / (new_stage[1] == 1).sum()
            new_err2 = ((new_stage[0] == 1) & (lbl == 1)).sum() / (new_stage[0] == 1).sum()
            one_list = [ori_err1, err1, new_err1, (add_stage == 1).sum(), ori_err2, err2, new_err2, (add_stage == 0).sum()]
            for ii in range(len(one_list)):
                sum_list[ii] += one_list[ii]
            k += 1
            print(name, 'fore, ori err:{:.3f}, add err:{:.3f}, new err:{:.3f}, add num{}'.format(ori_err1, err1, new_err1, (add_stage == 1).sum()))
            print(name, 'back, ori err:{:.3f}, add err:{:.3f}, new err:{:.3f}, add num{}'.format(ori_err2, err2, new_err2, (add_stage == 0).sum()))
            np.save(out_name, new_stage)
            np.save(out_w_name, add_stage_w)
        for ii in range(len(sum_list)):
            print(put_list[ii], sum_list[ii] / k)

if __name__ == '__main__':
    main()

