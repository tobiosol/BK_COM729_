
import os
import sys

from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))
subdirectory_path = os.path.join(current_dir, 'fundus_v2')
sys.path.append(subdirectory_path)

from fundus_v2.fundus_ftextractor import FundusFTExtractor
import proj_util
import os
import numpy as np
import torch
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

class FundusSVMClassifier:
    def __init__(self, cnn_models, val_loader, device):
        self.device = device
        self.val_loader = val_loader
        
        self.f_extractor = FundusFTExtractor(device=device, cnn_models=cnn_models)
        self.svm_model = make_pipeline(
            StandardScaler(),
            svm.SVC(C=1.0, gamma='scale', kernel='rbf', probability=True)
        )
        
    def train(self):
        features, labels = self.f_extractor.el_extract_features(dataloader=self.val_loader)
        
        # Apply SMOTE to the training data
        smote = SMOTE(random_state=42)
        features_resampled, labels_resampled = smote.fit_resample(features, labels)
        
        # Perform PCA on the resampled data
        n_components = min(features_resampled.shape[0], features_resampled.shape[1], 6)  # Adjust to your desired number of components
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(features_resampled)
        
        param_grid = {'svc__C': [0.1, 1, 10], 'svc__kernel': ['rbf', 'linear']}
        grid_search = GridSearchCV(self.svm_model, param_grid, cv=10)
        grid_search.fit(reduced_features, labels_resampled)
        
        self.svm_model = make_pipeline(StandardScaler(), grid_search.best_estimator_)
        
        # Fit the model on the entire resampled and reduced training set
        self.svm_model.fit(reduced_features, labels_resampled)
        
    def evaluate(self):
        features, labels = self.f_extractor.el_extract_features(dataloader=self.val_loader)
        predictions = self.svm_model.predict(features)
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
            fpr[i], tpr[i], _ = roc_curve(labels, self.svm_model.predict_proba(features)[:, i], pos_label=i)
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
        overall_roc_auc = roc_auc_score(labels, self.svm_model.predict_proba(features), average='macro', multi_class='ovr')
        print(f"Overall ROC AUC: {overall_roc_auc:.4f}")
