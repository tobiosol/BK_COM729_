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
from fundus_v2 import fundus_img_preprocessorV23
from fundus_v2 import fundus_img_augmentor, fundus_dataloader, fundus_cnn_module, fundus_cnn_trainer, fundus_img_dataset,cnn_model_wrapper, fundus_rf_classifier, fundus_svm_classifier



class MLEnsembleClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.augmentor = fundus_img_augmentor.FundusImageAugmentor()
        self.test_transform = self.augmentor.test_transform
        self.val_loader = self._get_val_loader()
        self.rf_classifier = None  # Initialize as None

    def _preprocess_image(self, pil_image):
        preprocessor = fundus_img_preprocessorV23.FundusImagePreprocessorV23()
        processed_image = preprocessor.predictor_preprocess(pil_image)
        processed_tensor_image = self.test_transform(processed_image)
        image_tensor = torch.tensor(np.array(processed_tensor_image), dtype=torch.float32).unsqueeze(0)
        return image_tensor

    def _get_val_loader(self):
        dataloader = fundus_dataloader.FundusDataLoader(image_dir=proj_util.VALIDATION_DIR,
                                                        csv_file=proj_util.VALIDATION_LABEL_PATH,
                                                        batch_size=32, transform=self.test_transform,
                                                        shuffle=False, train=False, num_augmentations=1)
        return dataloader.get_loader()

    def load_or_train_rf_model(self):
        if self.rf_classifier is None:
            cnn_models = [
                cnn_model_wrapper.CNNModelWrapper('densenet121'),
                cnn_model_wrapper.CNNModelWrapper('resnext50_32x4d')
            ]
            self.rf_classifier = fundus_rf_classifier.FundusRFClassifier(cnn_models=cnn_models,
                                                                        val_loader=self.val_loader,
                                                                        device=self.device)
            self.rf_classifier.load_model()  # Load the trained model if available

    def predict_rf(self, pil_image):
        self.load_or_train_rf_model()  # Ensure the model is loaded or trained
        processed_tensor_image = self._preprocess_image(pil_image=pil_image)
        
        feature_extractor = DenseNet121FeatureExtractor()
        extracted_features = feature_extractor.extract_features(processed_tensor_image)
        predicted_class, prediction_matrix = self.rf_classifier.predict(extracted_features)
        return predicted_class, prediction_matrix












import torch
from torchvision import models

class DenseNet121FeatureExtractor:
    def __init__(self, num_classes=3):
        self.model = models.densenet121(pretrained=True)
        self.model.features.conv0 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(num_ftrs, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(512, num_classes)
        )
        self.model.eval()  # Set to evaluation mode

    def extract_features(self, image_tensor):
        """
        Extract features from a tensor image.

        Args:
            image_tensor (torch.Tensor): Input image tensor (shape: [batch_size, 1, height, width])

        Returns:
            torch.Tensor: Extracted features (shape: [batch_size, num_features])
        """
        with torch.no_grad():
            features = self.model.features(image_tensor)
            # Flatten the features (keep batch dimension)
            features = features.view(features.size(0), -1)
        return features


















# Example usage:
ensemble_classifier = MLEnsembleClassifier()
pil_image = Image.open("timg/56.png").convert("RGB")
predicted_class, prediction_matrix = ensemble_classifier.predict_rf(pil_image)
print(f"Predicted class: {predicted_class}")
print(f"Prediction matrix: {prediction_matrix}")