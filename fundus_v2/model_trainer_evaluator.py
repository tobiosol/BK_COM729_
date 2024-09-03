from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

class ModelTrainerEvaluator:
    def __init__(self, models, train_loader, val_loader):
        self.models = models
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_and_evaluate(self):
        for name, model in self.models:
            print(f"Training {name}...")
            model.train()
            print(f"Evaluating {name}...")
            model.evaluate()
            
    
    # def _prepare_data(self, data_loader):
    #     features = []
    #     labels = []
    #     for images, lbls in data_loader:
    #         images = images.view(images.size(0), -1).numpy()  # Flatten images
    #         lbls = lbls.view(-1).numpy()  # Flatten labels
    #         features.append(images)
    #         labels.append(lbls)
    #     features = np.concatenate(features, axis=0)
    #     labels = np.concatenate(labels, axis=0)
    #     return features, labels

    # def _evaluate_model(self, model, name):
    #     val_features, val_labels = self._prepare_data(self.val_loader)
    #     predictions = model.predict(val_features)
    #     accuracy = accuracy_score(val_labels, predictions)
    #     print(f"{name} Validation Accuracy: {accuracy:.4f}")
    #     print("Confusion Matrix:\n", confusion_matrix(val_labels, predictions))
    #     print("Classification Report:\n", classification_report(val_labels, predictions))
