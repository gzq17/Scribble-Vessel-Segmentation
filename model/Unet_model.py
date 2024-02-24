import torch
from torch import nn
import math
import numpy as np
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch,eps=1e-3, momentum=0.01),
            )

    def forward(self, x):

        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch,eps=1e-3, momentum=0.01),
        )

    def forward(self, x):
        x = self.up(x)
        return x

class UnetOri(nn.Module):

    def __init__(self, in_ch=1, out_ch=2):
        super(UnetOri, self).__init__()

        filters = [64, 128, 256, 512, 1024]
        # filters = [32, 64, 128, 256, 512]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up4 = up_conv(filters[4], 4)
        self.Up_conv4 = conv_block(filters[-2] + 4, filters[3])


        self.Up3 = up_conv(filters[3], 4)
        self.Up_conv3 = conv_block(filters[-3] + 4, filters[2])

        self.orientation1 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                          conv_block(filters[2], 90),
                                          torch.nn.Softmax(dim=1))
        
        self.Up2 = up_conv(filters[2], filters[1])
        self.Up_conv2 = conv_block(filters[2], filters[1])

        self.Up1 = up_conv(filters[1], filters[0])
        self.Up_conv1 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.Norm = nn.BatchNorm2d(out_ch,eps=1e-3, momentum=0.01)

        self.active = torch.nn.Softmax(dim=1)

    def forward(self, tensor_list):
        #x = tensor_list.tensors
        x1 = tensor_list

        e1 = self.Conv1(x1)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d4 = self.Up4(e5)
        d4 = torch.cat((d4,e4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((d3,e3), dim=1)
        d3 = self.Up_conv3(d3)
        ori_out = self.orientation1(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((d2,e2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        d1 = torch.cat((d1,e1), dim=1)
        d1 = self.Up_conv1(d1)

        d0 = self.Conv(d1)
        norm_out = self.Norm(d0)
        out = self.active(norm_out)

        return out

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats 
        self.temperature = temperature 
        self.normalize = normalize  
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        mask = torch.zeros((x.shape[0], x.shape[2], x.shape[3]), device=x.device, dtype=torch.bool)
        not_mask = ~mask  
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)  
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t 
        pos_y = y_embed[:, :, :, None] / dim_t 
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class PositionEmbeddingLearned(nn.Module):
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(128, num_pos_feats)
        self.col_embed = nn.Embedding(128, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

def getGaussianKernel(ksize, sigma=0):
    if sigma <= 0:
        # 根据 kernelsize 计算默认的 sigma，和 opencv 保持一致
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8 
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center) # 元素与矩阵中心的横向距离
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2)) # 计算一维卷积核
    # 根据指数函数性质，利用矩阵乘法快速计算二维卷积核
    kernel = kernel1d[..., None] @ kernel1d[None, ...] 
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum() # 归一化
    return kernel

class EF_block(nn.Module):
    def __init__(self, in_ch, ksize=3):
        super(EF_block, self).__init__()
        self.conv1_1 = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.convTrans1 = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
        self.sigmaSpace = 0.15 * ksize + 0.35
        self.pad = (ksize - 1) // 2
        self.ksize = ksize
        self.weights_space = getGaussianKernel(ksize, self.sigmaSpace).cuda()
    
    def forward(self, x1, x2):
        x1 = self.conv1_1(x1)
        x1 = self.convTrans1(x1)
        batch_img_pad = F.pad(x1, pad=[self.pad, self.pad, self.pad, self.pad], mode='reflect')
        patches = batch_img_pad.unfold(2, self.ksize, 1).unfold(3, self.ksize, 1)
        patch_dim = patches.dim()
        diff_color = patches - x1.unsqueeze(-1).unsqueeze(-1)
        weights_color = torch.exp(-(diff_color ** 2) / (2 * self.sigmaSpace ** 2))
        weights_space_dim = (patch_dim - 2) * (1,) + (self.ksize, self.ksize)
        weights_space = self.weights_space.view(*weights_space_dim).expand_as(weights_color)
        weights = weights_space * weights_color
        weights_sum = weights.sum(dim=(-1, -2))
        weighted_pix = (weights * patches).sum(dim=(-1, -2)) / weights_sum
        weighted_pix = torch.sigmoid(weighted_pix)
        x = weighted_pix.expand(-1, x2.shape[1], -1, -1).mul(x2)
        x = x + x2
        return x

class UnetOriPositionBF(nn.Module):

    def __init__(self, in_ch=1, out_ch=2, embed='learn', N_steps=64):
        super(UnetOriPositionBF, self).__init__()
        filters = [32, 64, 128, 256, 512]
        if embed == 'sin':
            f_num = filters[2] * 2
            self.position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
        elif embed == 'learn':
            f_num = filters[2] * 2
            self.position_embedding = PositionEmbeddingLearned(N_steps)
        else:
            f_num = filters[2]
            self.position_embedding = None
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(f_num, filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])
        self.RE_block4 = EF_block(filters[4], 3)
        self.RE_block3 = EF_block(filters[3], 5)
        self.RE_block2 = EF_block(filters[2], 7)
        self.RE_block1 = EF_block(filters[1], 9)

        self.Up4 = up_conv(filters[4], 4)
        self.Up_conv4 = conv_block(260, filters[3])


        self.Up3 = up_conv(filters[3], 4)
        self.Up_conv3 = conv_block(132, filters[2])

        self.orientation1 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                          conv_block(filters[2], 90),
                                          torch.nn.Softmax(dim=1))
        
        self.Up2 = up_conv(filters[2], filters[1])
        self.Up_conv2 = conv_block(filters[2], filters[1])

        self.Up1 = up_conv(filters[1], filters[0])
        self.Up_conv1 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.Norm = nn.BatchNorm2d(out_ch,eps=1e-3, momentum=0.01)

        self.active = torch.nn.Softmax(dim=1)

    def forward(self, tensor_list):
        #x = tensor_list.tensors

        e1 = self.Conv1(tensor_list)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        x1 = self.RE_block1(e2, e1)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        x2 = self.RE_block2(e3, e2)

        e4 = self.Maxpool3(e3)
        if self.position_embedding is not None:
            pos = self.position_embedding(e4)
            e4 = torch.cat([e4, pos], dim=1)
        e4 = self.Conv4(e4)
        x3 = self.RE_block3(e4, e3)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        x4 = self.RE_block4(e5, e4)
        
        d4 = self.Up4(e5)
        d4 = torch.cat((d4,x4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((d3,x3), dim=1)
        d3 = self.Up_conv3(d3)
        ori_out = self.orientation1(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((d2,x2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        d1 = torch.cat((d1,x1), dim=1)
        d1 = self.Up_conv1(d1)

        d0 = self.Conv(d1)
        norm_out = self.Norm(d0)
        out = self.active(norm_out)

        return out

