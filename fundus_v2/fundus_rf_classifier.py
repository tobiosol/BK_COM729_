
import os
import sys
from joblib.externals.loky.backend.context import set_start_method
from sklearn.svm import SVC
import warnings


# Set the start method for multiprocessing
set_start_method('spawn')


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))
subdirectory_path = os.path.join(current_dir, 'fundus_v2')
sys.path.append(subdirectory_path)


from fundus_v2 import fundus_ftextractor
import proj_util
import os
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


class FundusRFClassifier:
    def __init__(self, cnn_models, val_loader, device):
        self.device = device
        self.val_loader = val_loader
        
        self.f_extractor = fundus_ftextractor.FundusFTExtractor(device=device, cnn_models=cnn_models)
        self.rf_model = make_pipeline(StandardScaler(), RandomForestClassifier())
        
        feature_names = [model.model_name for model in cnn_models]
        self.feature_name = "_".join(feature_names)
        self.features_file = f"{self.feature_name}_rf_features.npy"
        self.labels_file = f"{self.feature_name}_rf_labels.npy"
    
    def train(self):
        if os.path.exists(self.features_file) and os.path.exists(self.labels_file):
            # Load features and labels from file
            features = np.load(self.features_file)
            labels = np.load(self.labels_file)
            print("Loaded features and labels from file.")
        else:
            features, labels = self.f_extractor.el_extract_features(dataloader=self.val_loader)
            # Save features and labels to file
            np.save(self.features_file, features)
            np.save(self.labels_file, labels)
            print("Extracted and saved features and labels.")
        
        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=3, init='pca', learning_rate='auto')
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
        # self.print_metrics(labels=labels, predictions=predictions, features=features)
    
    def evaluate(self):
        features, labels = self.f_extractor.el_extract_features(dataloader=self.val_loader)
        features = TSNE(n_components=3, init='pca', learning_rate='auto').fit_transform(features)
        predictions = self.rf_model.predict(features)
        self.print_metrics(labels=labels, predictions=predictions, features=features)

    def print_metrics(self, labels, predictions, features):
        # Confusion Matrix
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        # Classification Report
        print("Classification Report:\n", classification_report(labels, predictions))
        print("Classification Matrix:\n", cm)

        # ROC Curve and AUC for each class
        n_classes = len(np.unique(labels))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(labels, self.rf_model.predict_proba(features)[:, i], pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curve for each class
        plt.figure(figsize=(10, 7))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (area = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

        # Calculate and print overall ROC AUC
        overall_roc_auc = roc_auc_score(labels, self.rf_model.predict_proba(features), average='macro', multi_class='ovr')
        print(f"Overall ROC AUC: {overall_roc_auc:.4f}")