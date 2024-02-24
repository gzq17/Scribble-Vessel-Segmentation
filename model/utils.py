import torch
import numpy as np
import copy
from scipy.ndimage import zoom,rotate


def convert_targets(targets, device):
    target_masks = targets["masks"]
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 3, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks

def save_on_master(*args, **kwargs):
    torch.save(*args, **kwargs)

def bound_check(x_b, x_e, len_, upper):
    if x_b < 0:
        x_b, x_e = 0, len_
    if x_e > upper:
        x_e, x_b = upper, upper - len_
    return x_b, x_e

def t2n(a):
    return (b.detach().cpu().numpy() for b in a)

def randomrotate2(samples2, target2, ori_mask_two2):
    rot_list = []
    for i in range(0, samples2.shape[0]):
        rot = 0
        if np.random.rand() < 0.85:
            rot = np.random.rand() * 360
            samples2[i, 0, :, :] = rotate(samples2[i, 0, :, :], rot, order=1, reshape=False, mode='constant',cval=0)
            target2[i, 0, :, :] = rotate(target2[i, 0, :, :], rot, order=0, reshape=False, mode='constant',cval=0)
            ori_mask_two2[i, 0, :, :] = rotate(ori_mask_two2[i, 0, :, :], rot, order=0, reshape=False, mode='constant',cval=0)
        rot_list.append(rot)
    return samples2, target2, ori_mask_two2, rot_list

def coronary_improve_cutmix(samples1, target, ori_mask_two, device):
    sz = 512
    indices = np.random.permutation(samples1.size(0))
    samples1, target, ori_mask_two = t2n([samples1, target, ori_mask_two])
    samples2, target2 = copy.deepcopy(samples1[indices]), copy.deepcopy(target[indices])
    ori_mask_two2 = copy.deepcopy(ori_mask_two[indices])
    samples2, target2, ori_mask_two2, rot_list = randomrotate2(samples2, target2, ori_mask_two2)
    samples_mix, target_mix = np.zeros(samples1.shape), np.zeros(target.shape)
    ori_mask_two_mix = np.zeros(ori_mask_two.shape)
    M1, M2 = np.zeros(samples1.shape), np.zeros(samples1.shape)
    for i in range(0, samples1.shape[0]):
        im1, tg1, mask_two1 = samples1[i, 0], target[i, 0], ori_mask_two[i, 0]
        im2, tg2, mask_two2 = samples2[i, 0], target2[i, 0], ori_mask_two2[i, 0]
        M1_one = np.zeros(im1.shape)
        M1_s = np.zeros((im1.shape[0] // 8, im1.shape[1] // 8))
        M2_one = np.zeros(im1.shape)
        if np.random.rand() < 0.85:
            if np.random.rand() < 0.75:
                index = np.where(tg2 == 1)
            else:
                index = np.where(mask_two2 == 1)
            if index[0].shape[0] <= 2:
                index = np.where(tg2 == 0)
            if index[0].shape[0] <= 2:
                index = np.where(tg2 == 2)
        else:
            tg_copy = copy.deepcopy(tg2)
            mm = np.ones(tg_copy.shape)
            mm[80:80+352, 80:80+352] = 0
            tg_copy[mm == 1] = -1
            index = np.where(tg_copy == 0)
            if index[0].shape[0] <= 2:
                index = np.where(tg_copy == 1)
            if index[0].shape[0] <= 2:
                index = np.where(tg_copy == -1)
            if index[0].shape[0] <= 2:
                index = np.where(tg_copy == 2)
        if index[0].shape[0] <= 1:
            xxxx, yyyy = 256, 256
        else:
            rand_i = np.random.randint(0, index[0].shape[0] - 1)
            xxxx, yyyy = index[0][rand_i], index[1][rand_i]
        x_center, y_center = round(xxxx / 8), round(yyyy / 8)
        x_len, y_len = 20, 20
        x_b, y_b = x_center - x_len // 2, y_center - y_len // 2
        x_e, y_e = x_b + x_len, y_b + y_len
        x_b, x_e = bound_check(x_b, x_e, x_len, im2.shape[-2] // 8)
        y_b, y_e = bound_check(y_b, y_e, y_len, im2.shape[-1] // 8)
        M1_one[x_b * 8: x_e * 8, y_b * 8: y_e * 8] = 1
        M1[i, 0] = M1_one
        M1_s[x_b:x_e, y_b:y_e] = 1
        samples_mix[i, 0] = im1 * (1 - M1_one) + im2 * M1_one
        target_mix[i, 0] = tg1 * (1 - M1_one) + tg2 * M1_one
        ori_mask_two_mix[i, 0] = mask_two1 * (1 - M1_one) + mask_two2 * M1_one
        xx_len, yy_len = np.random.randint(6, 20), np.random.randint(6, 20)
        xx_b, yy_b = np.random.randint(0, im1.shape[0] // 8 - xx_len - 1), np.random.randint(0, im1.shape[1] // 8 - yy_len - 1)
        M2_one[xx_b * 8: (xx_b + xx_len) * 8, yy_b * 8: (yy_b + yy_len) * 8] = 1
        rot = np.random.rand() * 360
        M2_one = rotate(M2_one, rot, order=0, reshape=False, mode='constant',cval=0)
        samples_mix[i, 0][M2_one == 1] = 0
        target_mix[i, 0][M2_one == 1] = 0
        ori_mask_two_mix[i, 0][M2_one == 1] = 0
        M2[i, 0] = M2_one
    return {'img_mix': torch.from_numpy(samples_mix).to(device).float(),
            'target_mix': torch.from_numpy(target_mix).to(device),
            'ori_mask_mix2': torch.from_numpy(ori_mask_two_mix).to(device),
            'M1_mask':M1,
            'M2_mask':M2,
            'rot_list':rot_list,
            'indices':indices}
