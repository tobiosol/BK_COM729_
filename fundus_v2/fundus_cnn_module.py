import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights, MobileNet_V2_Weights, ResNet18_Weights, Inception_V3_Weights

class FundusCNNModule(nn.Module):
    
    def __init__(self, base_model=None, num_classes=3):
        super(FundusCNNModule, self).__init__()
        self.num_classes = num_classes
        
        if base_model is None:
            self.base_model = self.build_simple_cnn()
        else:
            self.base_model = base_model
            self._modify_base_model()
        self.fc = nn.Linear(self.num_features, self.num_classes)
        self.dropout = nn.Dropout(0.5)

    def build_simple_cnn(self):
        model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 74 * 74, 128),  # Adjusted input size for 299x299 images
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.num_features = 128  # Set num_features for consistency with other models
        return model

    def _modify_base_model(self):
        if isinstance(self.base_model, models.VGG):
            self._modify_vgg()
        elif isinstance(self.base_model, models.MobileNetV3):
            self._modify_mobilenet()
        elif isinstance(self.base_model, models.ResNet):
            self._modify_resnet()
        elif isinstance(self.base_model, models.Inception3):
            self._modify_inception()
        elif isinstance(self.base_model, models.DenseNet):
            self._modify_densenet()
        elif isinstance(self.base_model, models.ViT):
            self._modify_vit()
        elif isinstance(self.base_model, models.ResNeXt):
            self._modify_resnext()
        elif isinstance(self.base_model, models.Xception):
            self._modify_xception()
        elif isinstance(self.base_model, models.NASNet):
            self._modify_nasnet()
        else:
            raise ValueError("Unsupported model type")

    def _modify_vgg(self):
        self.base_model.features = nn.Sequential()

        for layer in self.base_model.features:
            if isinstance(layer, nn.Conv2d):
                if layer.in_channels == 3:
                    continue
                
                layer.in_channels = 1
                layer.kernel_size = (5,)

            elif isinstance(layer, nn.MaxPool2d):
                layer.kernel_size = (2,)

        self.base_model.classifier = nn.Sequential(
            nn.Linear(10 * 9, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10)  # Replace with the actual number of classes
        )

    def _modify_mobilenet(self):
        self.num_features = self.base_model.classifier[-1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.num_features, self.num_classes)
        )
        self.base_model.features[0][0] = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        self._initialize_weights(self.base_model.features[0][0])

    def _modify_resnet(self):
        self.num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.num_features, self.num_classes)
        )
        self.base_model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self._initialize_weights(self.base_model.conv1)

    def _modify_inception(self):
        self.num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.num_features, self.num_classes)
        )
        self.base_model.Conv2d_1a_3x3.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2)
        self._initialize_weights(self.base_model.Conv2d_1a_3x3.conv)

    def _modify_densenet(self):
        self.num_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.num_features, self.num_classes)
        )
        self.base_model.features.conv0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self._initialize_weights(self.base_model.features.conv0)

    def _modify_vit(self):
        self.num_features = self.base_model.heads.head.in_features
        self.base_model.heads.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.num_features, self.num_classes)
        )

    def _modify_resnext(self):
        self.num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.num_features, self.num_classes)
        )
        self.base_model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self._initialize_weights(self.base_model.conv1)

    def _modify_xception(self):
        self.num_features = self.base_model.last_linear.in_features
        self.base_model.last_linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.num_features, self.num_classes)
        )
        self.base_model.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2)
        self._initialize_weights(self.base_model.conv1)

    def _modify_nasnet(self):
        self.num_features = self.base_model.last_linear.in_features
        self.base_model.last_linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.num_features, self.num_classes)
        )
        self.base_model.conv0 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=3, stride=2)
        self._initialize_weights(self.base_model.conv0)

    def _initialize_weights(self, layer):
        with torch.no_grad():
            layer.weight = nn.Parameter(layer.weight.mean(dim=1, keepdim=True))

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Example usage:
if __name__ == "__main__":
    
    base_model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    model = FundusCNNModule(base_model=base_model)
    
    # Print model parameters
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Parameter shape: {param.shape}")
        print(f"Parameter values: {param}\n")
    
    input_tensor = torch.randn(8, 1, 299, 299)  # Batch size of 8, 1 channel, 299x299 image
    output = model(input_tensor)
    print(output.shape)


# # Example usage
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = FundusClassifier(num_classes=3)  # Default to Simple CNN
# model.to(device)

# # Define the optimizer with weight decay (L2 regularization)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# # Define the loss function
# criterion = nn.CrossEntropyLoss()

# # Example input tensor with batch size of 8, single-channel, and dimensions 299x299
# input_tensor = torch.randn(8, 1, 299, 299).to(device)
# output = model(input_tensor)
# print(output)
