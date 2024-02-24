import cv2
import torch
from pathlib import Path
import copy
import numpy as np

def model_test2(args, model, dataloader_val, if_save=True):
    device = torch.device(args.device)
    result_out_path = args.output_dir + 'result2/'
    if result_out_path:
        Path(result_out_path).mkdir(parents=True, exist_ok=True)
    total_steps = len(dataloader_val)
    iterats = iter(dataloader_val)
    dice_sum, k = 0, 0
    Acc_sum, Sen_sum, Spe_sum = 0, 0, 0
    with torch.no_grad():
        for step in range(total_steps):
            data_one = next(iterats)
            samples = data_one['img'].to(device).float()
            target = data_one['lab'].to(device)
            outputs = model(samples)
            for i in range(0, outputs.shape[0]):
                src_masks = outputs[i:i+1]
                predict_p = src_masks[0][1]
                predict_p_np = copy.deepcopy(predict_p.detach().cpu().numpy())
                src_masks = src_masks.argmax(1)[0]
                targets_masks = target[i][0]
                FN = ((targets_masks == 1) & (src_masks == 0)).sum()
                FP = ((targets_masks == 0) & (src_masks == 1)).sum()
                TP = ((targets_masks == 1) & (src_masks == 1)).sum()
                TN = ((targets_masks == 0) & (src_masks == 0)).sum()
                Acc, Sen, Spe = (TP + TN) / (TP + TN + FP + FN), TP / (TP + FN), TN / (TN + FP)
                dice = 2 * TP / (2 * TP + FN + FP)
                Acc_sum += Acc * 100
                Sen_sum += Sen * 100
                Spe_sum += Spe * 100
                print(data_one['img_name'][i], '{:.2f}'.format(float(dice) * 100))
                dice_sum += float(dice)
                k += 1
                src_masks_n = src_masks.detach().cpu().numpy()
                src_masks_n[src_masks_n == 1] = 255
                if if_save:
                    cv2.imwrite(result_out_path + data_one['img_name'][i] + '.png', src_masks_n)
                    np.save(result_out_path + data_one['img_name'][i] + '.npy', predict_p_np)
    print(k)
    print('average dice:{:.4f}'.format(dice_sum / k * 100))
    print('average Acc:{:.4f}, Sen:{:.2f}, Spe:{:.2f}'.format(Acc_sum / k, Sen_sum / k, Spe_sum / k))
    
def pseudo_label(args, model, dataloader_val, if_save=True):
    device = torch.device(args.device)
    result_out_path = args.output_dir + 'stage1_res/'    
    Path(result_out_path).mkdir(parents=True, exist_ok=True)
    total_steps = len(dataloader_val)
    iterats = iter(dataloader_val)
    dice_sum, k = 0, 0
    Acc_sum, Sen_sum, Spe_sum = 0, 0, 0
    a_s, b_s, c_s, d_s, b_re_o_s, b_re_s, f_re_o_s, f_re_s = 0, 0, 0, 0, 0, 0, 0, 0
    f_s = 0
    with torch.no_grad():
        for step in range(total_steps):
            data_one = next(iterats)
            samples = data_one['img'].to(device).float()
            target = data_one['lab'].to(device)
            outputs = model(samples)
            for i in range(0, outputs.shape[0]):
                src_masks_n = outputs[i:i+1]
                result = copy.deepcopy(src_masks_n[0].detach().cpu().numpy())
                target_n = copy.deepcopy(target[i][0].detach().cpu().numpy())
                # print(result[0]+result[1])
                b_re_o, b_re = result[0] > 0.5, result[0] > 0.95
                f_re_o, f_re = result[1] >= 0.5, result[1] > 0.995# 0.9975#995
                ori_m = result[1] > 0.02
                a = ((b_re_o == 1) & (target_n == 1)).sum() / b_re_o.sum()
                b = ((b_re == 1) & (target_n == 1)).sum() / b_re.sum()
                c = ((f_re_o == 1) & (target_n == 0)).sum() / f_re_o.sum()
                d = ((f_re == 1) & (target_n == 0)).sum() / f_re.sum()
                f = ((ori_m == 0) & (target_n == 1)).sum() / target_n.sum()
                stage1_res = np.zeros((3, *target_n.shape))
                stage1_res[0][b_re] = 1
                stage1_res[1][f_re] = 1
                stage1_res[2][ori_m] = 1
                f_s += f
                a_s, b_s, c_s, d_s = a_s + a, b_s + b, c_s + c, d_s + d
                b_re_o_s, b_re_s, f_re_o_s, f_re_s = b_re_o_s + b_re_o.sum(), b_re_s + b_re.sum(), f_re_o_s + f_re_o.sum(), f_re_s + f_re.sum()
                ss = 'back ori err:{:.3f}, back now err:{:.3f}'.format(a, b)
                ss += ';fore ori err:{:.3f}, fore now err:{:.3f}'.format(c, d)
                print(ss)

                src_masks = src_masks_n.argmax(1)[0]
                targets_masks = target[i][0]
                FN = ((targets_masks == 1) & (src_masks == 0)).sum()
                FP = ((targets_masks == 0) & (src_masks == 1)).sum()
                TP = ((targets_masks == 1) & (src_masks == 1)).sum()
                TN = ((targets_masks == 0) & (src_masks == 0)).sum()
                Acc, Sen, Spe = (TP + TN) / (TP + TN + FP + FN), TP / (TP + FN), TN / (TN + FP)
                Acc_sum += Acc * 100
                Sen_sum += Sen * 100
                Spe_sum += Spe * 100
                dice=(2*torch.sum((src_masks==1)*(targets_masks==1)).float())/(torch.sum(src_masks==1).float()+torch.sum(targets_masks==1).float()+1e-10)
                # print(data_one['img_name'][i], '{:.2f}'.format(float(dice) * 100))
                dice_sum += float(dice)
                k += 1
                src_masks_n = src_masks.detach().cpu().numpy()
                src_masks_n[src_masks_n == 1] = 255
                if if_save:
                    np.save(result_out_path + data_one['img_name'][i] + '.npy', stage1_res)
    print(k)
    print('average dice:{:.4f}'.format(dice_sum / k * 100))
    print('average Acc:{:.4f}, Sen:{:.2f}, Spe:{:.2f}'.format(Acc_sum / k, Sen_sum / k, Spe_sum / k))
    ss = 'ave back ori err:{:.4f}, back now err:{:.4f}'.format(a_s / k, b_s / k)
    ss += ';fore ori err:{:.3f}, fore now err:{:.3f}'.format(c_s / k, d_s / k)
    print(ss)
    ss = 'back ori sum:{:.1f}, back now sum:{:.1f}'.format(b_re_o_s / k, b_re_s / k)
    ss += '; fore ori sum:{:.1f}, fore now sum:{:.1f}'.format(f_re_o_s / k, f_re_s / k)
    print(ss)
    print(f_s / k)

