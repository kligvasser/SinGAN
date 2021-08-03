import torch
from utils.core import imresize

class ConsistencyLoss(torch.nn.Module):
    def __init__(self, scale=0.5, criterion=torch.nn.MSELoss()):
        super(ConsistencyLoss, self).__init__()
        self.scale = scale
        self.criterion = criterion

    def forward(self, inputs, targets):
        targets = imresize(targets, scale=self.scale)
        inputs = imresize(inputs, scale=self.scale)

        loss = self.criterion(inputs, targets.detach())

        return loss
