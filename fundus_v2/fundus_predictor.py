import torch
import numpy as np
from torchvision import transforms

from fundus_v2 import fundus_img_preprocessorV23

class LiverOcularDiseasePredictor:
    
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Define class labels based on your index mapping
        self.class_labels = ["Cotton-wool Spots", "Diabetic Retinopathy", "No Identifiable Eye Disease"]
        
        self.test_transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def preprocess_image(self, pil_image):
        preprocessor = fundus_img_preprocessorV23.FundusImagePreprocessorV23()
        processed_image = preprocessor.predictor_preprocess(pil_image)
        processed_image = self.test_transform(processed_image)
        return processed_image
    
    
    
    def predict_single_image(self, image):
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

        predicted_class = self.class_labels[preds.item()]
        return predicted_class, probs.cpu().numpy()