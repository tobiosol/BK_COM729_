
import os
import sys

import numpy as np
from sklearn.model_selection import GridSearchCV

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))
subdirectory_path = os.path.join(current_dir, 'fundus_v2')
sys.path.append(subdirectory_path)

import proj_util

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn import svm
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import make_pipeline
from fundus_v2 import fundus_ftextractor
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
import proj_util
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


class FundusSVMClassifier:
    def __init__(self, cnn_models, train_loader, val_loader, device):
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.f_extractor = fundus_ftextractor.FundusFTExtractor(device=device, cnn_models=cnn_models, train_loader=train_loader, val_loader=val_loader)
        
        self.svm_model = make_pipeline(
            StandardScaler(),
            svm.SVC(C=1.0, gamma='scale', kernel='rbf')
        )
        
    def train(self):
        features, labels = self.f_extractor.el_extract_features(dataloader=self.train_loader)
        
        # Apply SMOTE to the training data
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        features_resampled, labels_resampled = smote.fit_resample(features, labels)
        
        # Perform PCA on the resampled data
        pca = PCA(n_components=6)  # Adjust to your desired number of components
        reduced_features = pca.fit_transform(features_resampled)
        
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(self.svm_model, param_grid, cv=10)
        grid_search.fit(reduced_features, labels_resampled)
        
        self.svm_model = make_pipeline(StandardScaler(), grid_search.best_estimator_)
        
        # Fit the model on the entire resampled and reduced training set
        self.svm_model.fit(features_resampled, labels_resampled)
        
    def evaluate(self):
        features, labels = self.f_extractor.extract_features(dataloader=self.val_loader)
        predictions = self.svm_model.predict(features)
        self.print_metrics(labels=labels, predictions=predictions)

    @staticmethod
    def print_metrics(labels, predictions):
        print("Confusion Matrix:\n", confusion_matrix(labels, predictions))
        print("Classification Report:\n", classification_report(labels, predictions))
        print(f"Accuracy: {accuracy_score(labels, predictions):.4f}")
        print(f"Precision: {precision_score(labels, predictions, average='weighted'):.4f}")
        print(f"Recall: {recall_score(labels, predictions, average='weighted'):.4f}")
        print(f"F1 Score: {f1_score(labels, predictions, average='weighted'):.4f}")
