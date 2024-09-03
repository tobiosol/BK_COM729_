
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from joblib.externals.loky.backend.context import set_start_method
from sklearn.svm import SVC


# Set the start method for multiprocessing
set_start_method('spawn')


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))
subdirectory_path = os.path.join(current_dir, 'fundus_v2')
sys.path.append(subdirectory_path)


from fundus_v2 import fundus_ftextractor
import proj_util
import torch
import torch.nn as nn
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

class FundusRFClassifier:
    def __init__(self, cnn_models, train_loader, val_loader, device):
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.f_extractor = fundus_ftextractor.FundusFTExtractor(device=device, cnn_models=cnn_models, train_loader=self.train_loader, val_loader=self.val_loader)
        self.rf_model = make_pipeline(StandardScaler(), RandomForestClassifier())
        
        feature_names = [model.model_name for model in cnn_models]
        
        print("_".join(feature_names))
        
        self.feature_name = proj_util.get_trained_model("_".join(feature_names))
        self.features_file = f"{self.feature_name}_rf_features.npy"
        self.labels_file = f"{self.feature_name}_rf_labels.npy"
    
    def train(self):
        if os.path.exists(self.features_file) and os.path.exists(self.labels_file):
            # Load features and labels from file
            features = np.load(self.features_file)
            labels = np.load(self.labels_file)
            print("Loaded features and labels from file.")
        else:
            dataloaders = {'train': self.train_loader}
            self.f_extractor.train_ensemble(num_epochs=10)
            
            features, labels = self.f_extractor.el_extract_features(dataloader=self.train_loader)
            # Save features and labels to file
            np.save(self.features_file, features)
            np.save(self.labels_file, labels)
            print("Extracted and saved features and labels.")
        
        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=3)
        features = tsne.fit_transform(features)
        
        # Perform grid search for optimal hyperparameters
        param_grid = {
            'randomforestclassifier__n_estimators': [10, 50, 100],
            'randomforestclassifier__max_depth': [None, 5, 10],
            'randomforestclassifier__min_samples_split': [2, 5, 10],
            'randomforestclassifier__min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(self.rf_model, param_grid, cv=StratifiedKFold(n_splits=5), n_jobs=2, verbose=2)
        grid_search.fit(features, labels)
    
        # Get the best model and fit it on the entire training set
        self.rf_model = grid_search.best_estimator_
        self.rf_model.fit(features, labels)
    
        predictions = self.rf_model.predict(features)
        self.print_metrics(labels=labels, predictions=predictions)
    
    def evaluate(self):
        features, labels = self.f_extractor.el_extract_features(dataloader=self.val_loader)
        features = TSNE(n_components=3).fit_transform(features)
        predictions = self.rf_model.predict(features)
        self.print_metrics(labels=labels, predictions=predictions)

    def print_metrics(self, labels, predictions):
        print("Confusion Matrix:\n", confusion_matrix(labels, predictions))
        print("Classification Report:\n", classification_report(labels, predictions))
