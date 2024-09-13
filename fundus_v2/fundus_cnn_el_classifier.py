import os
import sys
import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))
subdirectory_path = os.path.join(current_dir, 'fundus_v2')
sys.path.append(subdirectory_path)

import proj_util
from fundus_v2 import cnn_model_wrapper, fundus_img_augmentor, fundus_img_dataset
from fundus_v2 import fundus_img_preprocessorV23

class CNNEnsembleClassifier:
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        resnext_wrapper = cnn_model_wrapper.CNNModelWrapper('resnext50_32x4d')
        resnext_model_path = proj_util.get_trained_model(f"{resnext_wrapper.model_name}_model.pth")
        print(resnext_model_path)
        self.resnext_model = resnext_wrapper.model
        self.load_model(self.resnext_model, resnext_model_path)
        self.resnext_model.eval()
        self.resnext_model.to(self.device)
        
        
        densenet_wrapper = cnn_model_wrapper.CNNModelWrapper('densenet121')
        densenet_model_path = proj_util.get_trained_model(f"{densenet_wrapper.model_name}_model.pth")
        print(densenet_model_path)
        self.densenet_model = densenet_wrapper.model
        self.load_model(self.densenet_model, densenet_model_path)
        self.densenet_model.eval()
        self.densenet_model.to(self.device)
        
        self.test_transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
    def load_model(self, model, model_path):
        model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        print(f'Model loaded from {model_path}')
        return model
    
    def predict(self, image_tensor):
        image_tensor = image_tensor.to(self.device)  # Ensure input tensor is on the same device
        
        with torch.no_grad():
            resnext_output = self.resnext_model(image_tensor)
            densenet_output = self.densenet_model(image_tensor)

        # Combine predictions (e.g., averaging logits)
        ensemble_output = (resnext_output + densenet_output) / 2
        _, predicted_class = ensemble_output.max(1)

        return predicted_class.item(), ensemble_output.cpu().numpy()
    
    def preprocess_image(self, pil_image):
        preprocessor = fundus_img_preprocessorV23.FundusImagePreprocessorV23()
        processed_image = preprocessor.predictor_preprocess(pil_image)
        processed_tensor_image = self.test_transform(processed_image)
        # image_tensor = torch.tensor(np.array(processed_tensor_image), dtype=torch.float32).squeeze()
        # image_tensor = image_tensor.repeat(1, 1, 1)  # Repeat along the batch dimension
        image_tensor = torch.tensor(np.array(processed_tensor_image), dtype=torch.float32).unsqueeze(0)
        
        return image_tensor
    



# ensemble_classifier = CNNEnsembleClassifier()
# pil_image = Image.open("timg/56.png").convert("RGB")
# processed_tensor_image = ensemble_classifier.preprocess_image( pil_image=pil_image)
# predicted_class, prediction_matrix = ensemble_classifier.predict(processed_tensor_image)
# print(f"Predicted class: {predicted_class}")
# print(f"Prediction matrix: {prediction_matrix}")