import torch
import torch.nn as nn
import torchvision.models as models

class CNNModelWrapper:
    
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
    
    def VGG(self):
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, self.num_classes)
        )
        return model

    def ResNeXt50_32x4d(self):
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, self.num_classes)
        )
        return model

    def densenet121(self):
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, self.num_classes)
        )
        return model

    def ViT(self):
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.conv_proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
        num_ftrs = model.heads.head.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, self.num_classes)
        )
        return model