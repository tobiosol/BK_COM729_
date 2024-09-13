from itertools import cycle
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))
subdirectory_path = os.path.join(current_dir, 'fundus_v2')
sys.path.append(subdirectory_path)

from sklearn.calibration import label_binarize
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, train_test_split
from sklearn.metrics import accuracy_score, auc, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from fundus_v2 import cnn_model_wrapper, fundus_img_augmentor, fundus_img_dataset
from fundus_v2.early_stopping import EarlyStopping
import proj_util
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler
import seaborn as sns

class FundusCNNTrainer:

    def __init__(self, model_name, train_loader, val_loader):
        self.model_name = model_name
        self.model_path = proj_util.get_trained_model(f"{model_name}_model.pth")        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Class distribution
        num_samples = 930
        class_counts = [90, 290, 595]  #  CWS: 0, DR: 1, NI: 2
        # Calculate class weights
        class_weights = [num_samples / count for count in class_counts]
        self.class_weights_tensor = torch.tensor(class_weights).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
        
        self.early_stopper = EarlyStopping(patience=10)
        
        self.best_accuracy = 0
        self.learning_rates = []
        
        
        start_time = time.time()
        self.model = cnn_model_wrapper.CNNModelWrapper(self.model_name).model
        end_time = time.time()
        print(f'Time taken to initialize model: {(end_time - start_time):.2f} seconds')
        self.model.to(self.device)
        
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            print("No pre-trained model found. Training a new model.")
    
    
    
    
    def get_optimizer(self, lr, weight_decay, betas):
        print("get_optimizer model_name", self.model_name)
    
        if self.model_name == "densenet121":
            optimizer = torch.optim.Adam([
                {'params': self.model.model.features.parameters(), 'lr': lr, 'weight_decay': weight_decay},
                {'params': self.model.model.classifier.parameters(), 'lr': lr, 'weight_decay': weight_decay}
            ], betas=betas)
        elif self.model_name == "resnext50_32x4d":
            optimizer = torch.optim.Adam([
                {'params': self.model.model.fc.parameters(), 'lr': lr, 'weight_decay': weight_decay},
                {'params': [param for name, param in self.model.model.named_parameters() if "fc" not in name], 'lr': lr, 'weight_decay': weight_decay}
            ], betas=betas)
        elif self.model_name == "vgg":
            optimizer = torch.optim.Adam([
                {'params': self.model.model.features.parameters(), 'lr': lr, 'weight_decay': weight_decay},
                {'params': self.model.model.classifier.parameters(), 'lr': lr, 'weight_decay': weight_decay}
            ], betas=betas)
        elif self.model_name == "convnext_tiny":
            optimizer = torch.optim.Adam([
                {'params': self.model.model.classifier.parameters(), 'lr': lr, 'weight_decay': weight_decay},
                {'params': [param for name, param in self.model.model.named_parameters() if "classifier" not in name], 'lr': lr, 'weight_decay': weight_decay}
            ], betas=betas)
        else:
            raise ValueError("Invalid model name")
        return optimizer
    
    
    def compute_class_weights(self, dataloader, device, num_classes):
        all_labels = []
        # Iterate through the dataloader to collect all labels
        for data in dataloader:
            _, labels = data
            all_labels.extend(labels.cpu().numpy())
    
        # Convert labels to a numpy array
        all_labels = np.array(all_labels)
        
        # Debug: Print collected labels and their distribution
        # print("Collected labels:", all_labels)
        unique, counts = np.unique(all_labels, return_counts=True)
        print("Class distribution:", dict(zip(unique, counts)))
    
        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=all_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        
        # Debug: Print computed class weights
        print("Computed class weights:", class_weights)
    
        return class_weights

    def train(self, num_epochs):
        if os.path.exists(self.model_path):
            print("Model already trained. Skipping training.")
            start_time = time.time()
            self.load_model()
            end_time = time.time()
            print(f'Time taken to load model from folder: {(end_time - start_time):.2f} seconds')
            return
        
        start_time = time.time()
        self.optimizer = self.get_optimizer(lr=0.001, betas=(0.5, 0.99), weight_decay=1e-05)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f'Time taken to complete other initialization: {epoch_duration:.2f} seconds')
    
        for epoch in range(num_epochs):
            start_time = time.time()
    
            train_loss, train_acc, train_metrics = self.train_one_epoch(self.train_loader)
            val_loss, val_acc, val_metrics = self.validate(self.val_loader)
            self.scheduler.step(val_loss)
    
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
    
            end_time = time.time()
            epoch_duration = end_time - start_time
    
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
            print(f'ROC AUC: {val_metrics["roc_auc"]:.4f}')
            print(f'Current Learning Rate: {current_lr:.6f}')
            print(f'Time taken for epoch: {epoch_duration:.2f} seconds')
    
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.save_model()
    
            self.early_stopper(val_loss)
            if self.early_stopper.early_stop:
                print("Early stopping")
                break
    
        self.plot_learning_rates()
        
    
    
    def train_one_epoch(self, train_loader, threshold=0.5, num_classes=3):
        class_weights = self.compute_class_weights(train_loader, self.device, num_classes)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        self.model.to(self.device)
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
    
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels = labels.view(-1)
    
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
    
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        metrics = self.calculate_metrics(all_labels, all_preds)
        
        return epoch_loss, epoch_acc, metrics

    
    def validate(self, val_loader, num_classes=3):
        class_weights = self.compute_class_weights(val_loader, self.device, num_classes)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.model.eval()  # Set model to evaluation mode
        all_preds = []
        all_labels = []
        all_probs = []
        running_loss = 0.0
    
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())  # Apply softmax to get probabilities
    
        # Ensure y_true and y_score have the correct shape
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
    
        # Ensure the number of columns in all_probs matches the number of unique classes in all_labels
        num_classes = len(np.unique(all_labels))
        if all_probs.shape[1] != num_classes:
            print(f"Warning: Number of classes in y_true ({num_classes}) not equal to the number of columns in 'y_score' ({all_probs.shape[1]})")
            # Handle this case appropriately, e.g., by skipping this batch or adjusting the probabilities
    
        # Calculate metrics
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovo')
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        metrics = self.calculate_metrics(all_labels, all_preds)
    
        val_metrics = {
            'confusion_matrix': cm,
            'roc_auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'sensitivity': metrics["sensitivity"]
        }
        # self.plot_metrics(val_metrics, all_labels, all_probs)
        return val_loss, val_acc, val_metrics

    
    
    def calculate_metrics(self, labels, preds):
        precision = precision_score(labels, preds, average='weighted', zero_division=0)
        recall = recall_score(labels, preds, average='weighted', zero_division=0)
        f1 = f1_score(labels, preds, average='weighted', zero_division=0)
        
        # Calculate sensitivity (recall for the positive class)
        cm = confusion_matrix(labels, preds)
        sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    
        return {'precision': precision, 'recall': recall, 'f1': f1, 'sensitivity': sensitivity}
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f'Model saved to {self.model_path}')
    
    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.to(self.device)
        print(f'Model loaded from {self.model_path}')
    
    def plot_learning_rates(self):
        plt.plot(self.learning_rates)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate over Epochs')
        plt.show()

    def get_labels_for_kfold(self, dataset):
        labels = []
        for i in range(len(dataset.image_paths)):
            label = dataset.get_labels(dataset.image_paths[i])
            labels.append(label.item())

        return labels

    
    
    
    
    
    
    

    def create_balanced_batches(self, subset, labels, batch_size=3):
        # Count the number of labels passed
        label_counts = {label: labels.count(label) for label in set(labels)}
        print(f"Label counts: {label_counts}")
        
        # Create a mapping from labels to indices
        label_to_indices = {label: np.where(np.array(labels) == label)[0].tolist() for label in set(labels)}
        for label in label_to_indices:
            random.shuffle(label_to_indices[label])
        # print(f"Label to indices: {label_to_indices}")
        
        batches = []
        while all(len(indices) >= 1 for indices in label_to_indices.values()):
            batch = []
            for label in label_to_indices:
                if len(label_to_indices[label]) > 0:
                    batch.append(label_to_indices[label].pop())
            if len(batch) < batch_size:
                additional_samples = random.sample(sum(label_to_indices.values(), []), batch_size - len(batch))
                batch.extend(additional_samples)
            batches.append(batch[:batch_size])  # Ensure batch size is exactly 3
            # print(f"Batch: {batch}")
        
        # Handle remaining indices
        remaining_indices = sum(label_to_indices.values(), [])
        if remaining_indices:
            random.shuffle(remaining_indices)
            for i in range(0, len(remaining_indices), batch_size):
                batch = remaining_indices[i:i + batch_size]
                if len(batch) < batch_size:
                    additional_samples = random.sample(remaining_indices, batch_size - len(batch))
                    batch.extend(additional_samples)
                batches.append(batch[:batch_size])  # Ensure batch size is exactly 3
                # print(f"Remaining batch: {batch}")
        
        # Ensure all batches have equal dimensions
        batches = [batch for batch in batches if len(batch) == batch_size]
        
        # Print the labels of the final batches
        final_batches_labels = [[labels[idx] for idx in batch] for batch in batches]
        unique_labels_in_batches = [set(batch_labels) for batch_labels in final_batches_labels]
        # print(f"Final batches: {batches}")
        # print(f"Final batches labels: {final_batches_labels}")
        print(f"Unique labels in final batches: {unique_labels_in_batches}")
        
        return batches
    
    
    
    
    
    
    
    
    
    def mod_kfold(self, dataset, num_epochs=10, n_splits=10):
        augmentor = fundus_img_augmentor.FundusImageAugmentor()
        train_transform = augmentor.train_transform
        test_transform = augmentor.test_transform
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = []
        labels = self.get_labels_for_kfold(dataset)
        
        print("labels", len(labels))
        print("dataset", len(dataset.image_paths))
        
        best_score = float('inf')
        for fold, (train_idx, val_idx) in enumerate(skf.split(dataset.image_paths, labels)):
            
            print('----------------------------------------------')
            print('----------------------------------------------')
            print(f'Fold {fold+1}/{n_splits}')
            
            train_dataset = fundus_img_dataset.FundusImageDataset(
                image_dir=dataset.image_dir,
                csv_file=dataset.csv_file,
                transform=train_transform,
                train=True,
                num_augmentations=dataset.num_augmentations
            )

            val_dataset = fundus_img_dataset.FundusImageDataset(
                image_dir=dataset.image_dir,
                csv_file=dataset.csv_file,
                transform=test_transform,
                train=False,
                num_augmentations=1
            )

            # Ensure validation set has at least one sample from each class
            val_labels = [val_dataset.get_labels(val_dataset.image_paths[idx]).item() for idx in val_idx]
            unique_classes = np.unique(labels)
            for cls in unique_classes:
                if cls not in val_labels:
                    # Find an index of the missing class in the training set
                    missing_class_idx = next(i for i, label in enumerate(labels) if label == cls and i in train_idx)
                    # Move the sample from training to validation
                    train_idx = list(train_idx)
                    val_idx = list(val_idx)
                    train_idx.remove(missing_class_idx)
                    val_idx.append(missing_class_idx)

            train_subset = Subset(train_dataset, train_idx)
            val_subset = Subset(val_dataset, val_idx)

            # Debug: Print indices and class distributions
            print(f"Train indices: {train_idx}")
            print(f"Validation indices: {val_idx}")
            train_labels = [train_dataset.get_labels(train_dataset.image_paths[idx]).item() for idx in train_idx]
            val_labels = [val_dataset.get_labels(val_dataset.image_paths[idx]).item() for idx in val_idx]
            print(f"Training set class distribution: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
            print(f"Validation set class distribution: {dict(zip(*np.unique(val_labels, return_counts=True)))}")

            # Calculate class weights for the training subset
            class_sample_counts = np.bincount(train_labels)
            weights = 1. / class_sample_counts
            samples_weights = weights[train_labels]
            train_sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

            train_loader = DataLoader(train_subset, batch_size=32, sampler=train_sampler)
            val_batches = self.create_balanced_batches(val_subset, val_labels, batch_size=32)
            val_loader = DataLoader(val_subset, batch_sampler=val_batches)

            # Debug: Print class distributions in val_loader
            val_loader_labels = []
            for data in val_loader:
                _, labels = data
                val_loader_labels.extend(labels.cpu().numpy())
            print(f"Validation loader class distribution: {dict(zip(*np.unique(val_loader_labels, return_counts=True)))}")

            self.optimizer = self.get_optimizer(lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999))
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)
            
            for epoch in range(num_epochs):
                train_loss, train_acc, train_metrics = self.train_one_epoch(train_loader, num_classes=3)
                val_loss, val_acc, val_metrics = self.validate(val_loader, num_classes=3)
                
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)

                # Handle missing 'f1' key
                val_metrics.setdefault('f1', 0.0)

                # Log metrics
                print(f'Epoch {epoch+1}/{num_epochs}:')
                print(f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}')
                print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
                print(f'Current Learning Rate: {current_lr:.6f}')

                if val_loss < best_score:
                    best_score = val_loss

                if val_acc > self.best_accuracy:
                    self.best_accuracy = val_acc
                    self.save_model()

                self.early_stopper(val_loss)
                if self.early_stopper.early_stop:
                    print("Early stopping")
                    break

            fold_results.append((train_loss, val_loss, val_metrics))

        # Calculate mean F1 score and other average metrics
        mean_f1_score = np.mean([result[2]['f1'] for result in fold_results])
        avg_val_loss = np.mean([result[0] for result in fold_results])
        avg_val_acc = np.mean([result[1] for result in fold_results])
        avg_val_metrics = {
            'precision': np.mean([result[2]['precision'] for result in fold_results]),
            'recall': np.mean([result[2]['recall'] for result in fold_results]),
            'f1_score': mean_f1_score
        }

        print(f"Mean F1 Score: {mean_f1_score}")
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        print(f'Average Validation Accuracy: {avg_val_acc:.4f}')
        print(f'Average Validation Metrics: {avg_val_metrics}')
        self.plot_learning_rates()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def grid_search_kfold(self, dataset, num_epochs, n_splits=15):
        augmentor = fundus_img_augmentor.FundusImageAugmentor()
        train_transform = augmentor.train_transform
        test_transform = augmentor.test_transform
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = []
        labels = self.get_labels_for_kfold(dataset)
        
        print("labels", len(labels))
        print("dataset", len(dataset.image_paths))
        
        param_grid = {
            'lr': [0.001, 0.0001, 0.00001],
            'batch_size': [16, 32, 64],
            'weight_decay': [1e-4, 1e-5, 1e-6],
            'betas': [(0.9, 0.999), (0.95, 0.999), (0.9, 0.95)]
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(dataset.image_paths, labels)):
            
            print('----------------------------------------------')
            print('----------------------------------------------')
            print('----------------------------------------------')
            print('----------------------------------------------')
            print(f'Fold {fold+1}/{n_splits}')
            
            train_dataset = fundus_img_dataset.FundusImageDataset(
                image_dir=dataset.image_dir,
                csv_file=dataset.csv_file,
                transform=train_transform,
                train=True,
                num_augmentations=dataset.num_augmentations
            )
    
            val_dataset = fundus_img_dataset.FundusImageDataset(
                image_dir=dataset.image_dir,
                csv_file=dataset.csv_file,
                transform=test_transform,
                train=False,
                num_augmentations=1
            )
    
            train_subset = Subset(train_dataset, train_idx)
            val_subset = Subset(val_dataset, val_idx)
    
            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    
            # model = cnn_model_wrapper.CNNModelWrapper(self.model_name).model
            # self.model = model
            # self.model.to(self.device)
            
            
            # Grid search for hyperparameters
            best_params = None
            best_score = float('inf')
            
            for lr in param_grid['lr']:
                for batch_size in param_grid['batch_size']:
                    for weight_decay in param_grid['weight_decay']:
                        for betas in param_grid['betas']:
                            
                            self.optimizer = self.get_optimizer(lr, weight_decay, betas)
                            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)
                            
                            for epoch in range(num_epochs):
                                train_loss, train_acc, train_metrics = self.train_one_epoch(train_loader)
                                val_loss, val_acc, val_metrics = self.validate(val_loader)
                                
                                self.scheduler.step(val_loss)
                                current_lr = self.optimizer.param_groups[0]['lr']
                                self.learning_rates.append(current_lr)
    
                                # Handle missing 'f1' key
                                val_metrics.setdefault('f1', 0.0)
    
                                # Log metrics
                                print(f'Epoch {epoch+1}/{num_epochs}:')
                                print(f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}')
                                print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
                                print(f'Current Learning Rate: {current_lr:.6f}')
    
                                if val_loss < best_score:
                                    best_score = val_loss
                                    best_params = {'lr': lr, 'batch_size': batch_size, 'weight_decay': weight_decay, 'betas': betas}
    
                                if val_acc > self.best_accuracy:
                                    self.best_accuracy = val_acc
                                    self.save_model()
    
                                self.early_stopper(val_loss)
                                if self.early_stopper.early_stop:
                                    print("Early stopping")
                                    break
    
            print(f'Best Parameters: {best_params}')
            fold_results.append((train_loss, val_loss, val_metrics))
    
        # Calculate mean F1 score and other average metrics
        mean_f1_score = np.mean([result[2]['f1'] for result in fold_results])
        avg_val_loss = np.mean([result[0] for result in fold_results])
        avg_val_acc = np.mean([result[1] for result in fold_results])
        avg_val_metrics = {
            'precision': np.mean([result[2]['precision'] for result in fold_results]),
            'recall': np.mean([result[2]['recall'] for result in fold_results]),
            'f1_score': mean_f1_score
        }
    
        print(f"Mean F1 Score: {mean_f1_score}")
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        print(f'Average Validation Accuracy: {avg_val_acc:.4f}')
        print(f'Average Validation Metrics: {avg_val_metrics}')
        self.plot_learning_rates()
        
        
        
        

        
    def print_formatted_metrics(self, val_metrics):
        print("Validation Metrics:")
        print("-------------------")
        for metric, value in val_metrics.items():
            if metric == 'confusion_matrix':
                print(f"{metric}:\n{value}")
            else:
                print(f"{metric}: {value:.4f}")
                
                

    
    
    def plot_roc_curve(self, model, val_loader):
        model.eval()
        all_labels = []
        all_probs = []
    
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
    
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
    
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
    
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
    
    
    
    
    # n_classes = len(np.unique(all_labels))  # Number of classes
    def plot_multiclass_roc_curve(self, all_labels, all_probs, n_classes):
        # Binarize the labels for multi-class ROC
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
    
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
    
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
        # Plot all ROC curves
        plt.figure()
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(i, roc_auc[i]))
    
        plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
                label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
    
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic for Multi-Class')
        plt.legend(loc="lower right")
        plt.show()

# # Debugging function to print model attributes
# def print_model_attributes(model):
#     print("Model attributes:")
#     for attr in dir(model):
#         if not attr.startswith('_'):
#             print(attr)


    
    
    
    def plot_metrics(self, val_metrics, all_labels, all_probs):
        # Plot Confusion Matrix
        cm = val_metrics['confusion_matrix']
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(all_labels), yticklabels=np.unique(all_labels))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
    
        # Plot ROC Curve
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(len(np.unique(all_labels))):
            fpr[i], tpr[i], _ = roc_curve(all_labels, all_probs[:, i], pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.figure(figsize=(10, 7))
        for i in range(len(np.unique(all_labels))):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
    
        # # Plot Validation Loss
        # plt.figure(figsize=(10, 7))
        # plt.plot(history['loss'], label='Training Loss')
        # plt.plot(history['val_loss'], label='Validation Loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.title('Training and Validation Loss')
        # plt.legend()
        # plt.show()
        
        
    


from torch.utils.data import Sampler
import random

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.labels = [dataset.dataset.get_labels(dataset.dataset.image_paths[idx]).item() for idx in dataset.indices]
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0].tolist() for label in set(self.labels)}
        for label in self.label_to_indices:
            random.shuffle(self.label_to_indices[label])
        self.batches = self.create_batches()

    def create_batches(self):
        batches = []
        while len(self.label_to_indices[0]) >= self.batch_size // len(self.label_to_indices):
            batch = []
            for label in self.label_to_indices:
                batch.extend(self.label_to_indices[label][:self.batch_size // len(self.label_to_indices)])
                self.label_to_indices[label] = self.label_to_indices[label][self.batch_size // len(self.label_to_indices):]
            batches.append(batch)
        return batches

    def __iter__(self):
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


# if __name__ == "__main__":
#     trainer = FundusCNNTrainer(model_name="convnext_tiny", train_loader=None, val_loader=None)
    
#     model = trainer.model
    # model = cnn_model_wrapper.DenseNet121Model(num_classes=3)
    # print(model)
    # print_model_attributes(model)

    # # Check if the classifier attribute exists
    # if hasattr(model.model.classifier, 'classifier'):
    #     print("Classifier attribute exists in the model.")
        
    # else:
    #     print("Classifier attribute does not exist in the model.")
    
    
    # optimizer = trainer.get_optimizer()
    
    """
    # Create stratified subsets
            train_subset_indices, _ = train_test_split(train_idx, test_size=0.0, stratify=[labels[i] for i in train_idx])
            val_subset_indices, _ = train_test_split(val_idx, test_size=0.0, stratify=[labels[i] for i in val_idx])
    
            train_subset = Subset(train_dataset, train_subset_indices)
            val_subset = Subset(val_dataset, val_subset_indices)
    
    """