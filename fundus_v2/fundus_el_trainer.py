import os
import sys






sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))
subdirectory_path = os.path.join(current_dir, 'fundus_v2')
sys.path.append(subdirectory_path)


import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.model_selection import LeaveOneOut
from fundus_v2 import fundus_el_model

from sklearn.model_selection import StratifiedKFold
from fundus_v2 import cnn_model_wrapper, early_stopping
import proj_util
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD
early_stopping = early_stopping.EarlyStopping(patience=10, min_delta=0.01)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

N_WORKERS = os.cpu_count()
class FundusEnsembleModelTrainer:
    
    def __init__(self, cnn_models, train_loader, val_loader):
        self.dataloaders = {'train': train_loader, 'val': val_loader}
        self.lr_list = []
        LR_FOUND = 1e-3
        
        feature_names = [model.model_name for model in cnn_models]
        print("_".join(feature_names))
        self.model_path = proj_util.get_trained_model("_".join(feature_names))
        
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.ensemble_model = fundus_el_model.FundusEnsembleModel(cnn_models)
        class_weights = self.compute_class_weights(train_loader=train_loader, device=self.device)
        
        lr_params = []
        if hasattr(self.ensemble_model, 'features'):
            lr_params.append({'params': self.ensemble_model.features.parameters(), 'lr': LR_FOUND / 10})
        if hasattr(self.ensemble_model, 'classifier'):
            lr_params.append({'params': self.ensemble_model.classifier.parameters(), 'lr': LR_FOUND})
        else:
            lr_params.append({'params': self.ensemble_model.parameters(), 'lr': LR_FOUND})
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(lr_params, weight_decay=1e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2, min_lr=1e-7)
        self.best_accuracy = 0
        
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            print("No pre-trained model found. Training a new model.")
    
    def compute_class_weights(self, train_loader, device):
        all_labels = torch.cat([labels.to(device) for _, labels in train_loader])
        all_labels_np = all_labels.cpu().numpy()
    
        class_weights = compute_class_weight('balanced', classes=np.unique(all_labels_np), y=all_labels_np)
    
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
        return class_weights
    
    def train_model(self, num_epochs=10):
        for epoch in range(num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.ensemble_model.train()
                else:
                    self.ensemble_model.eval()
    
                running_loss = 0.0
                running_corrects = 0
    
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
    
                    self.optimizer.zero_grad()
    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.ensemble_model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
    
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
    
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders[phase].dataset)
    
                print(f'Epoch {epoch}/{num_epochs - 1}, {phase} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
    
                if phase == 'val':
                    self.scheduler.step(epoch_loss)
                    self.lr_list.append(self.optimizer.param_groups[0]['lr'])
                    if epoch_acc > self.best_accuracy:
                        self.best_accuracy = epoch_acc
                        self.save_model()
        # # Plot the learning rate
        # plt.plot(range(num_epochs), self.lr_list)
        # plt.xlabel('Epoch')
        # plt.ylabel('Learning Rate')
        # plt.title('Learning Rate over Epochs')
        # plt.show()
    def save_model(self):
        torch.save(self.ensemble_model.state_dict(), self.model_path)
    
    def load_model(self):
        self.ensemble_model.load_state_dict(torch.load(self.model_path))
        self.ensemble_model.to(self.device)
        
    # def plot_learning_rates(self):
    #     plt.plot(self.learning_rates)
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Learning Rate')
    #     plt.title('Learning Rate over Epochs')
    #     plt.show()