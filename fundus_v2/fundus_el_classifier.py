import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

class FundusEnsembleClassifier:
    def __init__(self, models, train_loader=None, val_loader=None, device='cpu'):
        self.models = models
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device) if isinstance(device, str) else device
        self.ensemble_model = VotingClassifier(estimators=[(f'model_{i}', model) for i, model in enumerate(models)], voting='soft')
        
    def train(self):
        features, labels = self._prepare_data(self.train_loader)
        self.ensemble_model.fit(features, labels)
        print("Ensemble model trained successfully.")
        
    def evaluate(self):
        val_features, val_labels = self._prepare_data(self.val_loader)
        predictions = self.ensemble_model.predict(val_features)
        accuracy = accuracy_score(val_labels, predictions)
        print(f"Validation Accuracy: {accuracy:.4f}")
        return accuracy
    
    def _prepare_data(self, data_loader):
        features = []
        labels = []
        for images, lbls in data_loader:
            images = torch.tensor(images).to(self.device)
            model_features = [model(images) for model in self.models]
            combined_features = torch.cat(model_features, dim=1).cpu().numpy()
            features.append(combined_features)
            labels.append(lbls.cpu().numpy())
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        return features, labels