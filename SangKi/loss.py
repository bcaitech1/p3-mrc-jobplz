import torch.nn as nn
import torch.nn.functional as F
import torch
class Cross_FocalLoss(nn.Module):
    def __init__(self, weight=None,
    gamma=2., reduction='mean', **kwargs):
        nn.Module.__init__(self)
        self.weight=weight
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs,targets):
        loss_fct_cross = nn.CrossEntropyLoss()
        loss_cross = loss_fct_cross(inputs, targets)
        loss_fct_focal = FocalLoss()
        loss_focal = loss_fct_focal(inputs, targets)
        # 두 비율이 합쳐서 1이 되야함
        cross_rate = 0.75
        focal_rate = 0.25
        return loss_cross*cross_rate +loss_focal*focal_rate
        
class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights
    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight) 
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss