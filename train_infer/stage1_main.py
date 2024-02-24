import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys 
sys.path.append('/home/guozhanqiang/med27_code/scribbles/my_github_code')
import argparse
from model.parser import get_args_parser
from pathlib import Path
import torch
import numpy as np
import random
from model.MyDataset import DriveOriSeg
from model.Unet_model import UnetOri
from torch.utils.data import DataLoader
from model.SegmentationLoss import SetCriterion
from model.utils import convert_targets, save_on_master
from train_infer.model_test import model_test2, pseudo_label

def adjust_lr(optimizer, epoch, lr, args):
    lr_c = lr * ((1 - epoch/(args.epochs + 1)) ** 0.9)
    for p in optimizer.param_groups:
        p['lr'] = lr_c

def build_net(args):
    device = torch.device(args.device)
    model = UnetOri(args.in_channels, args.out_channels)
    model.to(device)
    weight_dict = {'Avg': args.Avg,
                   'loss_CrossEntropy': args.CrossEntropy_loss_coef}
    losses = ['CrossEntropy', 'Avg']
    criterion = SetCriterion(losses=losses, weight_dict=weight_dict, args=args)
    criterion.to(device)

    return model, criterion

def build_data(args, img_set='train/'):
    root_path = args.root_path + args.dataset + '/'
    img_folder, lab_folder = root_path + img_set + 'images/', root_path + img_set + 'labels/'
    train = True
    if 'TestSet' in img_set:
        train = False
    data_set = DriveOriSeg(img_folder, lab_folder, train=train)
    return data_set

def train_one_epoch(model, criterion, dataloader_dict, optimizer, device):
    model.train()
    criterion.train()
    total_steps = len(dataloader_dict)
    iterats = iter(dataloader_dict)
    loss_dic = {'seg_loss':0, 'all loss':0}
    for step in range(total_steps):
        data_one = next(iterats)
        samples = data_one['img'].to(device).float()
        target = data_one['lab'].to(device)
        targets_onehot = convert_targets({'masks':target}, device)
        outputs = model(samples)
        loss_dict = criterion(outputs, targets_onehot)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in ['loss_CrossEntropy'] if k in weight_dict)
        all_loss = losses
        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()
        loss_dic['seg_loss'] += float(losses)
        loss_dic['all loss'] += float(all_loss)
        if step % 5 == 0:
            print("step:{}, seg loss:{:.4f}".format(step, losses.item()))
    ss = ''
    for kk in loss_dic.keys():
        ss = ss + kk + ':{:.4f}  '.format(loss_dic[kk] / total_steps)
    print(ss)
    return ss

def val_one_epoch(model, criterion, dataloader_dict, device):
    model.eval()
    criterion.eval()
    total_steps = len(dataloader_dict)
    iterats = iter(dataloader_dict)
    k = 0
    loss_dic = {'seg_loss':0, 'all loss':0, 'dice':0}
    with torch.no_grad():
        for step in range(total_steps):
            data_one = next(iterats)
            samples = data_one['img'].to(device).float()
            target = data_one['lab'].to(device)
            targets_onehot = convert_targets({'masks':target}, device)
            outputs = model(samples)
            loss_dict = criterion(outputs, targets_onehot)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in ['loss_CrossEntropy'] if k in weight_dict)
            all_loss = losses 
            loss_dic['seg_loss'] += float(losses)
            loss_dic['dice'] += float(sum(loss_dict['Avg']))
            loss_dic['all loss'] += float(all_loss)
            k += len(loss_dict['Avg'])
    ss = ''
    for kk in loss_dic.keys():
        down = total_steps
        if kk == 'dice':
            down = k
        ss = ss + kk + ':{:.4f}  '.format(loss_dic[kk] / down)
    print(ss)
    return ss, loss_dic['dice'] / k

def train(args):
    args.output_dir = args.root_path + args.dataset + '_res/' + 'stage1_res/'
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    best_dice = 0
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)
    model, criterion = build_net(args)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    model_without_ddp = model
    param_dicts = [{"params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr, weight_decay=args.weight_decay)#weight_decay=args.weight_decay

    dataset_train_dict = build_data(img_set='train/', args=args)
    dataset_train_dict.__getitem__(0)
    print('Number of training images: {}'.format(dataset_train_dict.__len__()))
    dataset_val_dict = build_data(img_set='val/', args=args)
    print('Number of validation images: {}'.format(dataset_val_dict.__len__()))
    dataloader_train = DataLoader(dataset_train_dict, args.batch_size, shuffle=True, num_workers=4) 
    dataloader_val = DataLoader(dataset_val_dict, args.batch_size, shuffle=False, num_workers=4)

    loss_f = open(args.output_dir + 'loss.txt', 'w')
    for epoch in range(0, args.epochs):
        print(epoch)
        if (epoch + 1) % 10 == 0:
            adjust_lr(optimizer, epoch, args.lr, args)
        train_stats = train_one_epoch(model, criterion, dataloader_train, optimizer, device)
        # lr_scheduler.step()
        test_stats, dice_score = val_one_epoch(model, criterion, dataloader_val, device)
        loss_f.write(str(epoch).zfill(4) + ':\n')
        loss_f.write(train_stats + '\n')
        loss_f.write(test_stats + '\n\n')
        loss_f.flush()
        if args.output_dir:
            # checkpoint_paths = [output_dir / 'checkpoint.pth']
            checkpoint_paths = []
            if dice_score > best_dice and dice_score > 0.5:
                best_dice = dice_score
                print("Update best model!")
                checkpoint_paths.append(output_dir / 'best_checkpoint.pth')
                file_name = str(dice_score)[0:6]+'new_checkpoint.pth'
                checkpoint_paths.append(output_dir / file_name)
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1000 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            if len(checkpoint_paths) != 0:
                for checkpoint_path in checkpoint_paths:
                    save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
    loss_f.close()

def test_evaluate(args):
    args.full = False
    args.output_dir = args.root_path + args.dataset + '_res/' + 'stage1_res/'
    model, criterion = build_net(args)
    molde_name = args.output_dir + 'best_checkpoint.pth'
    para = torch.load(molde_name)
    print(para['epoch'])
    model.load_state_dict(para['model'])
    model.eval()

    dataset_test_dict = build_data(img_set='TestSet/', args=args)
    print('Number of training images: {}'.format(dataset_test_dict.__len__()))
    dataloader_test = DataLoader(dataset_test_dict, args.batch_size, shuffle=True, num_workers=4) 

    model_test2(args, model, dataloader_test, if_save=True)

def generate_pseudo(args):
    args.full = False
    args.output_dir = args.root_path + args.dataset + '_res/' + 'stage1_res/'
    model, criterion = build_net(args)
    molde_name = args.output_dir + 'best_checkpoint.pth'
    para = torch.load(molde_name)
    print(para['epoch'])
    model.load_state_dict(para['model'])
    model.eval()
    set_list = ['train/', 'val/', 'TestSet/']
    for img_set in set_list:
        root_path = args.root_path + args.dataset + '/'
        img_folder, lab_folder = root_path + img_set + 'images/', root_path + img_set + 'labels/'
        if img_set == 'train/':
            img_folder, lab_folder = root_path + img_set + 'images/', root_path + img_set + 'annotations/'
        dataset_test_dict = DriveOriSeg(img_folder, lab_folder, train=False)
        print('Number of training images: {}'.format(dataset_test_dict.__len__()))
        dataloader_test = DataLoader(dataset_test_dict, args.batch_size, shuffle=False, num_workers=4) 
        pseudo_label(args, model, dataloader_test, if_save=True)

def main():
    parser = argparse.ArgumentParser('REIVE', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args.mode, args.dataset)
    if args.mode == 'train':
        train(args)
    elif args.mode == 'infer':
        test_evaluate(args)
    elif args.mode == 'pseudo':
        generate_pseudo(args)
    # else:
    #     return 

if __name__ == '__main__':
    main()
