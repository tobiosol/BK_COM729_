import cv2
import numpy as np
import torch
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import torchvision.models as models
from torchvision.models import VGG16_Weights, MobileNet_V2_Weights, ResNet18_Weights, Inception_V3_Weights, ResNeXt50_32X4D_Weights, ViT_B_16_Weights

class ImageQualityChecker:
    def __init__(self, model_path=None):
        # Load a pre-trained model for incorrect label detection
        self.model = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT) if model_path is None else torch.load(model_path)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 3)
        )
        
        self.model.eval()
        
        # Define the transform for the model
        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Normalize(mean=0.5, std=0.5)
        ])
        
        # Set thresholds
        self.blurriness_threshold = 100.0
        self.noise_threshold = 50.0
        self.brightness_min = 50.0
        self.brightness_max = 200.0
    
    def detect_blurriness(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    
    
    
    
    def detect_noise(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise_level = np.std(gray)
        return noise_level
    
    def check_brightness(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        return brightness
    
    def detect_incorrect_labels(self, image, true_label):
        # if len(image.size) == 3:
        #     image = image.convert('L')

        # image_tensor = self.transform(image).unsqueeze(0)
        # if image_tensor.shape[1] != 1:
        #     image_tensor = image_tensor.permute(1, 0, 2, 3)
        
        
        # device = next(self.model.parameters()).device
        # image_tensor = image_tensor.to(device)
        
        
        # print("true_label ",true_label)        
        # with torch.no_grad():
        #     outputs = self.model(image_tensor)  
        
        # predicted_indices = torch.argmax(outputs, dim=1)
        # print("predicted_indices ",predicted_indices)      
        # predicted_index = predicted_indices.tolist()[0]
        
        # print("predicted_index ",predicted_index)
        # return predicted_index != true_label
        return False
    
    def automated_quality_check(self, image_path, true_label):
        image = cv2.imread(image_path)
        pil_image = Image.open(image_path)
        
        blurriness_score = self.detect_blurriness(image)
        noise_level = self.detect_noise(image)
        brightness = self.check_brightness(image)
        incorrect_label = self.detect_incorrect_labels(pil_image, true_label)
        
        quality_report = {
            'blurriness_score': blurriness_score,
            'noise_level': noise_level,
            'brightness': brightness,
            'incorrect_label': incorrect_label,
            'blurriness_issue': blurriness_score < self.blurriness_threshold,
            'noise_issue': noise_level > self.noise_threshold,
            'brightness_issue': not (self.brightness_min <= brightness <= self.brightness_max),
            'label_issue': incorrect_label
        }
        
        return quality_report

# # Example usage
# checker = ImageQualityChecker()
# image_path = '79.png'
# true_label = 1  # Example true label
# quality_report = checker.automated_quality_check(image_path, true_label)
# print(quality_report)
