import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class NoiseRobustLoss(_Loss):
    def __init__(self, alpha=0.1, beta=0.9, classes=2):
        super(NoiseRobustLoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.classes = classes
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, labels):
        N = pred.shape[0] * pred.shape[2] * pred.shape[3]
        
        pred_flat = pred.permute(0, 2, 3, 1).contiguous().view(N, self.classes)
        labels_flat = labels.contiguous().view(N).long()
        ce_per_pixel = self.ce(pred_flat, labels_flat)

        pred_softmax = F.softmax(pred_flat, dim=1)
        label_one_hot = F.one_hot(labels_flat, self.classes).float().to(pred.device)
        mae_per_pixel = 1.0 - torch.sum(pred_softmax * label_one_hot, dim=1)
        
        loss = self.alpha * ce_per_pixel.mean() + self.beta * mae_per_pixel.mean()
        
        return loss
    