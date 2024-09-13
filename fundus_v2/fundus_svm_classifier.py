
import os
import sys

from matplotlib import pyplot as plt
import numpy as np
from sklearn import ensemble
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

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

import joblib

class FundusSVMClassifier:
    def __init__(self, cnn_models, val_loader, device):
        self.device = device
        self.val_loader = val_loader
        
        self.f_extractor = FundusFTExtractor(device=device, cnn_models=cnn_models)
        self.num_models = 5  
        feature_names = [model.model_name for model in cnn_models]
        self.feature_name = "_".join(feature_names)
        self.svm_model_file = proj_util.get_trained_model(f"{self.feature_name}_svm_features.pkl")
        
        self.svm_classifiers = None
        self.ensemble_model = None
        
    
    def train_el(self):
        if os.path.exists(self.svm_model_file):
            print("Loaded existing model.")
            self.ensemble_model = joblib.load(self.svm_model_file)
        else:
            features, labels = self.f_extractor.el_extract_features(dataloader=self.val_loader)
            X_train, _, y_train, _ = train_test_split(features, labels, test_size=0.2, random_state=42)

            self.svm_classifiers = []
            for i in range(self.num_models):
                svm_model = make_pipeline(StandardScaler(), svm.SVC(C=1.0, gamma='scale', kernel='rbf'))
                svm_model.fit(X_train, y_train)
                self.svm_classifiers.append(svm_model)

            self.ensemble_model = VotingClassifier(estimators=self.svm_classifiers, voting='hard')
            joblib.dump(self.ensemble_model, self.svm_model_file)
            print("Trained and saved new model.")
    
    
    
    
    
    
    
    
    
    
        
        joblib.dump(self.ensemble_model, self.svm_model_file)

    def predict_el(self):
        # Predictions from the ensemble model
        predictions = self.ensemble_model.predict(self.features)
        return predictions

    def evaluate_el(self):
        features, labels = self.f_extractor.el_extract_features(dataloader=self.val_loader)
        
        predictions = self.ensemble_model.predict(features)
        accuracy = accuracy_score(labels, predictions)

        # Calculate additional metrics
        roc_auc = roc_auc_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='weighted')
        f1 = f1_score(labels, predictions, average='weighted')

        # Confusion matrix
        confusion = confusion_matrix(labels, predictions)
        print("Confusion Matrix:")
        print(confusion)

        print(f"Ensemble Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        
        

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
