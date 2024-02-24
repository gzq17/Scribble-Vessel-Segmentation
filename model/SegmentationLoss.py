import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SetCriterion(nn.Module):
    """ This class computes the loss for MMSCMR.
    """
    def __init__(self, losses, weight_dict, args):
        super().__init__()
        self.losses = losses
        self.weight_dict = weight_dict
        self.args = args

    def Avg(self, outputs, targets):
        src_masks = outputs.argmax(1)
        targets_masks = targets.argmax(1)
        i = 1
        dice=(2*torch.sum((src_masks==i)*(targets_masks==i),(1, 2)).float())/(torch.sum(src_masks==i,(1, 2)).float()+torch.sum(targets_masks==i,(1, 2)).float()+1e-10)
        return {"Avg": dice}

    def loss_CrossEntropy(self, outputs, targets, eps=1e-6):
        src_masks = outputs
        y_labeled = targets[:,0:2,:,:]
        cross_back = -(y_labeled[:, 0, :, :] * torch.log(src_masks[:, 0, :, :] + eps)).sum() / (y_labeled[:, 0, :, :]).sum()
        cross_fore = -(y_labeled[:, 1, :, :] * torch.log(src_masks[:, 1, :, :] + eps)).sum() / (y_labeled[:, 1, :, :]).sum()
        cross_entropy = cross_back + cross_fore
        losses = {"loss_CrossEntropy": cross_entropy}
        # cross_entropy = -torch.sum(y_labeled * torch.log(src_masks + eps), dim = 1)
        # losses = {"loss_CrossEntropy": cross_entropy.mean()}
        return losses
        
    def get_loss(self, loss, outputs, targets):
        loss_map = {'Avg': self.Avg, 
                    'CrossEntropy': self.loss_CrossEntropy}
        return loss_map[loss](outputs, targets)

    def forward(self, outputs, targets):
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))
        return losses