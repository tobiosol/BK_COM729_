from matplotlib import pyplot as plt
from fundus_v2 import fundus_el_model
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

class FundusFTExtractor:
    
    def __init__(self, device, cnn_models, train_loader, val_loader):
        
        self.device = device
        self.models = [model for model in cnn_models]
        
        
        self.ensemble_model = fundus_el_model.FundusEnsembleModel(self.models).to(device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.ensemble_model.parameters(), lr=1e-3, betas=(0.5, 0.99), weight_decay=1e-03)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        
        self.train_loader = train_loader       
        self.val_loader = val_loader
        
    def load_images(self, image_paths):
        images = []
        for path in image_paths:
            image = Image.open(path).convert('L')  # Convert to grayscale
            image = image.resize((224, 224))  # Resize to 224x224
            image = np.array(image).astype(np.float32)
            images.append(image)
        return np.array(images)

    def visualize_distributions(self, train_data, val_data):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(train_data.flatten(), bins=50, color='blue', alpha=0.7)
        plt.title('Training Data Distribution')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

        plt.subplot(1, 2, 2)
        plt.hist(val_data.flatten(), bins=50, color='green', alpha=0.7)
        plt.title('Validation Data Distribution')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    def normalize_datasets(self, train_dataset, val_dataset):
        train_images = self.load_images(train_dataset.image_paths)
        val_images = self.load_images(val_dataset.image_paths)
        
        train_mean = np.mean(train_images)
        train_std = np.std(train_images)
        val_mean = np.mean(val_images)
        val_std = np.std(val_images)

        train_dataset.images = (train_images - train_mean) / train_std
        val_dataset.images = (val_images - val_mean) / val_std

    def train_model(self, model, dataloaders, num_epochs=10):
        model = model.to(self.device)
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # print(f"Initial inputs.shape: {inputs.shape}")

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        # print(f"Outputs.shape: {outputs.shape}")
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        return model

    def train_ensemble(self, num_epochs=10):
        dataloaders = {'train': self.train_loader, 'val': self.val_loader}
        for model in self.models:
            self.train_model(model, dataloaders, num_epochs)


    def extract_features(self, dataloader):
        all_features = []
        all_labels = []
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                model_features = []
                for model in self.models:
                    outputs = model(inputs)
                    model_features.append(outputs.cpu().numpy())
                combined_features = np.concatenate(model_features, axis=1)
                all_features.append(combined_features)
                all_labels.append(targets.cpu().numpy())
        return np.concatenate(all_features), np.concatenate(all_labels)
    
    def el_extract_features(self, dataloader):
        features = []
        labels = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.ensemble_model(inputs)
                features.append(outputs.cpu().numpy())
                labels.append(targets.cpu().numpy())
                
        return np.concatenate(features), np.concatenate(labels)