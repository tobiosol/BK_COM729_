import torch
import numpy as np
from torch import nn
from torchvision import models
from torch.utils.data import DataLoader

class FundusEnsembleModel(nn.Module):
    def __init__(self, models):
        super(FundusEnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        outputs = torch.stack(outputs)
        outputs = torch.mean(outputs, dim=0)
        return outputs