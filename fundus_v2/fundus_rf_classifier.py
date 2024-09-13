
import os
import sys
from joblib.externals.loky.backend.context import set_start_method
from sklearn.svm import SVC
import warnings
from PIL import Image


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

from fundus_v2 import cnn_model_wrapper, fundus_img_augmentor, fundus_img_dataset
from fundus_v2 import fundus_img_preprocessorV23
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
        self.features_file = proj_util.get_trained_model(f"{self.feature_name}_rf_features.npy")
        self.labels_file = proj_util.get_trained_model(f"{self.feature_name}_rf_labels.npy")
        self.rf_file = proj_util.get_trained_model("rf_model.pth")
        
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
        self.save_model()
        
    
    
    def evaluate(self):
        self.load_model()
        features, labels = self.f_extractor.el_extract_features(dataloader=self.val_loader)
        features = TSNE(n_components=3, init='pca', learning_rate='auto').fit_transform(features)
        predictions = self.rf_model.predict(features)
        self.print_metrics(labels=labels, predictions=predictions, features=features)
    
    def save_model(self):
        torch.save(self.rf_model, self.rf_file)
        print(f"Random Forest model saved to {self.rf_file}")
    
    def load_model(self):
        if os.path.exists(self.rf_file):
            self.rf_model = torch.load(self.rf_file)
            print(f"Random Forest model loaded from {self.rf_file}")
        else:
            print(f"No saved model found. Training a new one...")
            self.train()
    
    
    def predict(self, extracted_features):
        
        try:
            # Standardize the features using StandardScaler
            standardized_features = self.rf_model.named_steps['standardscaler'].transform(extracted_features)
    
            # Predict using the random forest model
            prediction = self.rf_model.predict(standardized_features)
            prediction_probs = self.rf_model.predict_proba(standardized_features)
    
            return prediction.item(), prediction_probs
    
        except Exception as e:
            # Handle any exceptions gracefully (e.g., invalid input shape)
            print(f"Prediction error: {str(e)}")
            return None, None
        
        

    
    

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
        
        
    def preprocess_image(self, pil_image):
        preprocessor = fundus_img_preprocessorV23.FundusImagePreprocessorV23()
        processed_image = preprocessor.predictor_preprocess(pil_image)
        processed_tensor_image = self.test_transform(processed_image)
        # image_tensor = torch.tensor(np.array(processed_tensor_image), dtype=torch.float32).squeeze()
        # image_tensor = image_tensor.repeat(1, 1, 1)  # Repeat along the batch dimension
        image_tensor = torch.tensor(np.array(processed_tensor_image), dtype=torch.float32).unsqueeze(0)
        
        return image_tensor
        
        
# rf_classifier = FundusRFClassifier()
# pil_image = Image.open("timg/56.png").convert("RGB")
# processed_tensor_image = rf_classifier.preprocess_image(pil_image=pil_image)
# predicted_class, prediction_matrix = rf_classifier.predict(processed_tensor_image)
# print(f"Predicted class: {predicted_class}")
# print(f"Prediction matrix: {prediction_matrix}")








