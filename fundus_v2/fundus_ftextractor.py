import os
import sys
from matplotlib import pyplot as plt
from fundus_v2 import fundus_el_model
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))
subdirectory_path = os.path.join(current_dir, 'fundus_v2')
sys.path.append(subdirectory_path)


import proj_util

class FundusFTExtractor:
    
    def __init__(self, device, cnn_models):
        self.device = device
        self.models = [model for model in cnn_models]
        
        for model in self.models:
            model_path = proj_util.get_trained_model(f"{model.model_name}_model.pth")
            print(model_path)
            self.load_model(model.model, model_path)
            model.model.eval()
            model.to(device)
        
        self.ensemble_model = fundus_el_model.FundusEnsembleModel(self.models).to(device)


    def load_model(self, model, model_path):
        model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        print(f'Model loaded from {model_path}')
        return model

    def extract_features(self, dataloader):
        all_features = []
        all_labels = []
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                model_features = []
                for model in self.models:
                    outputs = model(inputs)
                    model_features.append(outputs.cpu().numpy())
                combined_features = np.concatenate(model_features, axis=1)
                all_features.append(combined_features)
                all_labels.append(targets.cpu().numpy())
        return np.concatenate(all_features), np.concatenate(all_labels)
    
    
    
    def el_extract_features(self, dataloader):
        features = []
        labels = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.ensemble_model(inputs)
                features.append(outputs.cpu().numpy())
                labels.append(targets.cpu().numpy())
                
        return np.concatenate(features), np.concatenate(labels)