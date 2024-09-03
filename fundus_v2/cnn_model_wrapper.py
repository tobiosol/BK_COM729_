import time
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.init as init

class AttentionLayer(nn.Module):
    def __init__(self, in_features):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 4),  # Adjusted to match the input features
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 4, in_features),  # Adjusted to match the input features
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.attention(x)

    
class DenseNet121Model(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(DenseNet121Model, self).__init__()
        self.model = models.densenet121(pretrained=True)
        self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.model.features.norm0 = nn.BatchNorm2d(64)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
        
        
        
        for param in self.model.parameters():
            param.requires_grad = True
        

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


class ResNeXt50_32x4dModel(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(ResNeXt50_32x4dModel, self).__init__()
        self.model = models.resnext50_32x4d(pretrained=True)
        
        # Modify the first convolutional layer to accept 1 channel input
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

        # Unfreeze more layers
        for param in self.model.parameters():
            param.requires_grad = True

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


# class DenseNet121Model(nn.Module):
#     def __init__(self, num_classes=3, dropout_rate=0.5):
#         super(DenseNet121Model, self).__init__()
#         self.model = models.densenet121(pretrained=True)
#         self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.model.features.norm0 = nn.BatchNorm2d(64)
#         num_ftrs = 50176  # Number of features after flattening
        
#         self.attention = AttentionLayer(num_ftrs)
#         self.model.classifier = nn.Sequential(
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(num_ftrs, 512),
#             nn.ReLU(),
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(512, num_classes)
#         )
#         self._initialize_weights()
        
#         for param in self.model.parameters():
#             param.requires_grad = True
        
#         # # # Freeze all layers except the last 5 layers
#         # for name, param in self.model.named_parameters():
#         #     if 'classifier' not in name:
#         #         param.requires_grad = False

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.model.features(x)  # Pass through the DenseNet features
#         x = torch.flatten(x, 1)  # Flatten the tensor
#         attention_weights = self.attention(x)  # Apply attention
#         x = x * attention_weights  # Element-wise multiplication with attention weights
#         x = self.model.classifier(x)  # Fully connected layers
#         return x

# # Example usage
# model = DenseNet121Model(num_classes=3)
# input_tensor = torch.randn(32, 1, 224, 224)
# output = model(input_tensor)
# print("Output shape:", output.shape)



# class ResNeXt50_32x4dModel(nn.Module):
#     def __init__(self, num_classes=3, dropout_rate=0.5):
#         super(ResNeXt50_32x4dModel, self).__init__()
#         self.model = models.resnext50_32x4d(pretrained=True)
        
#         # Modify the first convolutional layer to accept 1 channel input
#         self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         num_ftrs = self.model.fc.in_features
#         self.model.fc = nn.Sequential(
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(num_ftrs, 512),
#             nn.ReLU(),
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(512, num_classes),
#         )
#         self._initialize_weights()

#         # Unfreeze more layers
#         for param in self.model.parameters():
#             param.requires_grad = True

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         return self.model(x)















    
# class ResNeXt50_32x4dModel(nn.Module):
#     def __init__(self, num_classes=3, dropout_rate=0.5):
#         super(ResNeXt50_32x4dModel, self).__init__()
#         self.model = models.resnext50_32x4d(pretrained=True)
        
#         # Modify the first convolutional layer to accept 1 channel input
#         self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
#         num_ftrs = self.model.fc.in_features
        
#         self.model.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(num_ftrs, 512),  # Adjust input size
#             nn.ReLU(),
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(512, num_classes)
#         )
#         self._initialize_weights()

#         # # Freeze the early layers
#         # for name, param in self.model.named_parameters():
#         #     if name.startswith('conv1') or name.startswith('bn1') or name.startswith('layer1'):
#         #         param.requires_grad = False
#         #     else:
#         #         param.requires_grad = True

#         # Unfreeze more layers
#         for param in self.model.parameters():
#             param.requires_grad = True

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)

#     def features(self, x):
#         # Extract features from the model
#         x = self.model.conv1(x)
#         x = self.model.bn1(x)
#         x = self.model.relu(x)
#         x = self.model.maxpool(x)
#         x = self.model.layer1(x)
#         x = self.model.layer2(x)
#         x = self.model.layer3(x)
#         x = self.model.layer4(x)
#         x = self.model.avgpool(x)
#         x = torch.flatten(x, 1)
#         return x

#     def forward(self, x):
#         x = self.features(x)
#         x = self.model.fc(x)
#         return x




# # Example usage
# model = ResNeXt50_32x4dModel(num_classes=3)
# input_tensor = torch.randn(32, 1, 224, 224)
# output = model(input_tensor)
# print("Output shape:", output.shape)



class VGGModel(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(VGGModel, self).__init__()
        self.model = models.vgg16(pretrained=True)
        
        # Modify the first convolutional layer to accept 1 channel input
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.model.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            *list(self.model.features.children())[1:]
        )
        
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(25088, num_ftrs),  # Adjust input size to match the flattened tensor dimensions
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, num_classes)
        )
        
        self._initialize_weights()
        
        for param in self.model.parameters():
            param.requires_grad = True
        

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.model.features(x)
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = self.model.classifier(x)
        return x


















class ConvNeXtTinyModel(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(ConvNeXtTinyModel, self).__init__()
        self.model = models.convnext_tiny(pretrained=True)
        num_ftrs = self.model.classifier[2].in_features
        
        # Add dropout layers
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, num_ftrs),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, num_classes)
        )
        
        self._initialize_weights()

        # Unfreeze more layers
        for param in self.model.parameters():
            param.requires_grad = True

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)
    
    
class CNNModelWrapper(nn.Module):
    
    def __init__(self, model_name, num_classes=3):
        super(CNNModelWrapper, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        start_time = time.time()
        self.model = self._initialize_model()
        end_time = time.time()
        print(f'Time taken to _initialize_model: {(end_time - start_time):.2f} seconds')
    
    def _initialize_model(self):
        
        print("_initialize_model model_name", self.model_name)
        if self.model_name == "densenet121":
            return DenseNet121Model(num_classes=3)
        elif self.model_name == "resnext50_32x4d":
            return ResNeXt50_32x4dModel(num_classes=3)
        elif self.model_name == "vgg":
            return VGGModel(num_classes=3)
        elif self.model_name == "convnext_tiny":
            return ConvNeXtTinyModel(num_classes=3)
        else:
            raise ValueError("Invalid model name")
        
    def forward(self, x):
        return self.model(x)
    
    


# # Example usage
# model_wrapper = CNNModelWrapper('densenet121')
# input_tensor = torch.randn(1, 1, 224, 224)  # Example input
# output = model_wrapper(input_tensor)
# print(output)
