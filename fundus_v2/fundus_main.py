import os
import sys





sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))
subdirectory_path = os.path.join(current_dir, 'fundus_v2')
sys.path.append(subdirectory_path)



from fundus_v2 import fundus_el_model, fundus_el_trainer
from fundus_v2.fundus_ftextractor import FundusFTExtractor
import numpy as np
import torch
import proj_util
from fundus_v2 import fundus_img_augmentor, fundus_dataloader, fundus_cnn_module, fundus_cnn_trainer, fundus_img_dataset,cnn_model_wrapper, fundus_rf_classifier, fundus_svm_classifier, fundus_cnn_classifiers
from fundus_v2 import model_trainer_evaluator, fundus_el_classifier, lr_finder

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

augmentor = fundus_img_augmentor.FundusImageAugmentor()
train_transform = augmentor.train_transform
test_transform = augmentor.test_transform

dataloader = fundus_dataloader.FundusDataLoader(image_dir=proj_util.TRAINING_DIR, csv_file=proj_util.TRAIN_LABEL_PATH, 
                                                   batch_size=32, transform=train_transform, train=True, num_augmentations=5)
train_loader = dataloader.get_loader()

dataloader = fundus_dataloader.FundusDataLoader(image_dir=proj_util.VALIDATION_DIR, csv_file=proj_util.VALIDATION_LABEL_PATH, 
                                                   batch_size=32, transform=test_transform, shuffle=False, train=False, num_augmentations=1)
val_loader = dataloader.get_loader()

def extract_labels(data_loader):
    labels = []
    for _, lbls in data_loader:
        labels.append(lbls.cpu())  # Keep labels as tensors for potential later use
    return torch.cat(labels)

def extract_data_bell(data_loader):
    data = []
    for images, _ in data_loader:
        flattened_images = images.flatten(start_dim=1)  # Flatten on GPU
        data.append(flattened_images)
    return torch.cat(data)


def plot_class_distribution(labels):
    unique_labels = set(labels)
    num_classes = len(unique_labels)
    
    if num_classes > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(labels, bins=num_classes, edgecolor='k', alpha=0.7)
        plt.xlabel('Class')
        plt.ylabel('Frequency')
        plt.title('Class Distribution')
        plt.show()
    else:
        print("No classes found in the labels.")

def plot_class_distribution_seaborn(labels):
    df = pd.DataFrame({'Class': labels})
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution')
    plt.show()
    


def plot_bell_curve(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
    y = norm.pdf(x, mean, std_dev)
    
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, density=True, alpha=0.6, edgecolor='k')  # Removed color argument
    plt.plot(x, y, 'k', linewidth=2)
    plt.title('Bell Curve (Normal Distribution)')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()
    

# def display_value_counts(csv_file, value=1):
#     # Read the CSV file
#     data = pd.read_csv(csv_file)
    
#     # Exclude the first column (index 0)
#     data = data.iloc[:, 1:]
    
#     # Count the occurrences of the specified value in each column
#     value_counts = (data == value).sum()
    
#     # Plot the counts
#     plt.figure(figsize=(10, 6))
#     value_counts.plot(kind='bar', edgecolor='black')
#     plt.title(f'Distribution of Value {value} by Column')
#     plt.xlabel('Columns')
#     plt.ylabel('Count')
#     plt.grid(False)
#     plt.show()
    
    
def display_value_counts(csv_file, value=1):
    # Read the CSV file
    data = pd.read_csv(csv_file)
    
    # Exclude the first column (index 0)
    data = data.iloc[:, 1:]
    
    # Count the occurrences of the specified value in each column
    value_counts = (data == value).sum()
    
    # Plot the counts
    plt.figure(figsize=(10, 6))
    bars = plt.bar(value_counts.index, value_counts.values, edgecolor='black')
    plt.title(f'Distribution of Training Target Classes')
    plt.xlabel('Target Classes')
    plt.ylabel('Count')
    plt.grid(False)
    
    # Add count labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom')
    
    plt.show()


def main():
    # Initialize the CNN Model Wrapper and Ensemble Feature Extractor
    cnn_model_wrapper = fundus_cnn_classifiers.CNNModelWrapper()
    model_name_list = ['ResNeXt50_32x4d']
    cnn_models = [getattr(cnn_model_wrapper, model_name)() for model_name in model_name_list]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model in cnn_models:
        model.to(device)

    # Train and evaluate Random Forest
    rf_classifier = fundus_rf_classifier.FundusRFClassifier(cnn_models=cnn_models, train_loader=train_loader, val_loader=val_loader, device=device)
    rf_classifier.train()
    rf_classifier.evaluate()

    # # Train and evaluate SVM
    # svm_classifier = fundus_svm_classifier.FundusSVMClassifier(cnn_models=cnn_models, train_loader=train_loader, val_loader=val_loader, device=device)
    # svm_classifier.train()
    # svm_classifier.evaluate()


# optimizer = optim.AdamW(model.parameters(), lr=START_LR)
# 
def evaluateModels():

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # Initialize the CNNModelWrapper instances
    # cnn_models = [
    #     cnn_model_wrapper.CNNModelWrapper('densenet121'),
    #     cnn_model_wrapper.CNNModelWrapper('resnext50_32x4d')
    # ]

    # # # Initialize the FundusRFClassifier
    # # rf_classifier = fundus_rf_classifier.FundusRFClassifier(cnn_models=cnn_models, val_loader=val_loader, device=device)
    # # # Train and evaluate the classifier
    # # rf_classifier.train()
    # # rf_classifier.evaluate()
    
    
    # # # Train and evaluate SVM
    # svm_classifier = fundus_svm_classifier.FundusSVMClassifier(cnn_models=cnn_models, val_loader=val_loader, device=device)
    # # Train and evaluate the classifier
    # svm_classifier.train()
    # svm_classifier.evaluate()
    
    
    
    
    
    # cnn_models = [cnn_model_wrapper.CNNModelWrapper('resnext50_32x4d').model]
    # rf_classifier = fundus_rf_classifier.FundusRFClassifier(cnn_models=cnn_models, train_loader=train_loader, val_loader=val_loader, device=device)
    # # # svm_classifier = fundus_svm_classifier.FundusSVMClassifier(cnn_models=cnn_models, train_loader=train_loader, val_loader=val_loader, device=device)
    
    # models = [("Random Forest", rf_classifier)]
    # trainer_evaluator = model_trainer_evaluator.ModelTrainerEvaluator(models, train_loader, val_loader)
    # trainer_evaluator.train_and_evaluate()
    
    
    
    # restnet_cnn_trainer = fundus_cnn_trainer.FundusCNNTrainer(model_name='ResNeXt50_32x4d', train_loader=train_loader, val_loader=val_loader)
    # denset_cnn_trainer = fundus_cnn_trainer.FundusCNNTrainer(model_name='densenet121', train_loader=train_loader, val_loader=val_loader)

    # models = [("ResNeXt50_32x4d", restnet_cnn_trainer), ("densenet121", denset_cnn_trainer)]
    # # models = [("SVM", svm_classifier)]
    # models = [("Random Forest", rf_classifier), ("SVM", svm_classifier)]
    

    # Train and evaluate ensemble model
    # ensemble_classifier = fundus_el_classifier.FundusEnsembleClassifier(models=models, train_loader=train_loader, val_loader=val_loader)
    # ensemble_classifier.train()
    # ensemble_classifier.evaluate()
    
    
    
    # Initialize individual models
    # model_vgg = cnn_model_wrapper.CNNModelWrapper('VGG').to(device)
    # model_resnext = cnn_model_wrapper.CNNModelWrapper('ResNeXt50_32x4d').to(device)
    # model_densenet = cnn_model_wrapper.CNNModelWrapper('densenet121').to(device)
    
    
    # cnn_models = [cnn_model_wrapper.CNNModelWrapper('densenet121').model().to(device), 
    #               cnn_model_wrapper.CNNModelWrapper('resnext50_32x4d').model().to(device)]
        
    # ensemble_model_trainer = fundus_el_trainer.FundusEnsembleModelTrainer(cnn_models, train_loader=train_loader, val_loader=val_loader)
    # trained_model = ensemble_model_trainer.train_model()
    
    
    # MobileNetV2
    # 77%
    # cnn_trainer = fundus_cnn_trainer.FundusCNNTrainer(model_name='densenet121', train_loader=train_loader, val_loader=val_loader)
    # cnn_trainer.train(num_epochs=10)
    # val_loss, val_acc, val_metrics = cnn_trainer.validate(val_loader=val_loader)
    # print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
    # cnn_trainer.print_formatted_metrics(val_metrics)
    
    
    # lrs, losses = cnn_trainer.lr_range_test()
    # cnn_trainer.plot_lr_vs_loss(lrs, losses)
    
    # print(f'Train Metrics: {self.print_formatted_metrics(train_metrics)}')
    # print(f'Val Metrics: {cnn_trainer.print_formatted_metrics(val_metrics)}')
    
    # cnn_trainer = fundus_cnn_trainer.FundusCNNTrainer(model_name='densenet121', train_loader=train_loader, val_loader=val_loader)
    # cnn_trainer.train(num_epochs=15)
    # val_loss, val_acc, val_metrics = cnn_trainer.validate(val_loader=val_loader)
    # print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
    # # print(f'Train Metrics: {self.print_formatted_metrics(train_metrics)}')
    # print(f'Val Metrics: {cnn_trainer.print_formatted_metrics(val_metrics)}')
    
    # model = cnn_model_wrapper.CNNModelWrapper('IMAGENET1K_V1').to(device)
    # lr_f = lr_finder.LRFinder(model, device)
    # lr_f.range_test(train_loader, end_lr=10, num_iter=100)
    # lr_f.plot()
    
    
    
    # # cnn_trainer.train_model(num_epochs=5)



    dataset = fundus_img_dataset.FundusImageDataset(
                image_dir=proj_util.TRAINING_DIR,
                csv_file=proj_util.TRAIN_LABEL_PATH,
                transform=None,
                train=True,
                num_augmentations=1
            )
    
    cnn_trainer = fundus_cnn_trainer.FundusCNNTrainer(model_name='resnext50_32x4d', train_loader=train_loader, val_loader=val_loader)
    cnn_trainer.train(num_epochs=15)
    # # cnn_trainer.mod_kfold(dataset, num_epochs=10, n_splits=10)
    # # cnn_trainer.grid_search_kfold(dataset, num_epochs=10, n_splits=10)
    # # cnn_trainer.train_kfold(dataset, num_epochs=24, n_splits=15)
    # # cnn_trainer.train_loocv(dataset, num_epochs=5)
    val_loss, val_acc, val_metrics = cnn_trainer.validate(val_loader)
    print(f'Validation Loss: {val_loss:.4f} Validation Accuracy: {val_acc:.4f}')
    print(f'Validation Metrics: {cnn_trainer.print_formatted_metrics(val_metrics)}')
    
    # cnn_trainer = fundus_cnn_trainer.FundusCNNTrainer(model_name='ViT', train_loader=train_loader, val_loader=val_loader)
    # # cnn_trainer.train_kfold(dataset, num_epochs=10, n_splits=10)
    # cnn_trainer.train(num_epochs=5)
    # val_loss, val_acc, val_metrics = cnn_trainer.validate(val_loader)
    # print(f'Validation Loss: {val_loss:.4f} Validation Accuracy: {val_acc:.4f}')
    # print(f'Validation Metrics: {cnn_trainer.print_formatted_metrics(val_metrics)}')
    
    # cnn_trainer = fundus_cnn_trainer.FundusCNNTrainer(model_name='nas', train_loader=train_loader, val_loader=val_loader)
    # cnn_trainer.train_kfold(dataset, num_epochs=25, n_splits=10)
    # val_loss, val_acc, val_metrics = cnn_trainer.validate(val_loader)
    # print(f'Validation Loss: {val_loss:.4f} Validation Accuracy: {val_acc:.4f}')
    # print(f'Validation Metrics: {cnn_trainer.print_formatted_metrics(val_metrics)}')
    
    # cnn_trainer = fundus_cnn_trainer.FundusCNNTrainer(model_name='simple', train_loader=train_loader, val_loader=val_loader)
    # cnn_trainer.train_kfold(dataset, num_epochs=25, n_splits=10)
    # val_loss, val_acc, val_metrics = cnn_trainer.validate(val_loader)
    # print(f'Validation Loss: {val_loss:.4f} Validation Accuracy: {val_acc:.4f}')
    # print(f'Validation Metrics: {cnn_trainer.print_formatted_metrics(val_metrics)}')
    
    

if __name__ == "__main__":
    # evaluateModels()
    # labels = extract_labels(train_loader)
    # plot_class_distribution(labels)
    # plot_class_distribution_seaborn(labels=labels)
    display_value_counts(proj_util.TRAIN_LABEL_PATH)
    # display_value_counts(proj_util.VALIDATION_LABEL_PATH)
    
    # data = extract_data_bell(train_loader)
    # plot_bell_curve(data)


# if __name__ == "__main__":
#     main()

    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(models[0].parameters(), lr=0.001, momentum=0.9)  # Adjust as needed
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    extractor = FundusFTExtractor(device='cuda', model_list=models)
    extractor.train_ensemble(dataloaders, criterion, optimizer, num_epochs=25)
    # Extract features
    features, labels = extractor.extract_features(test_dataloader)
    """