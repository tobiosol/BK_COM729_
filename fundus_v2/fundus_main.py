import os
import sys



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))
subdirectory_path = os.path.join(current_dir, 'fundus_v2')
sys.path.append(subdirectory_path)

from fundus_v2 import fundus_cnn_el_classifier, fundus_el_model, fundus_el_trainer
from fundus_v2.fundus_ftextractor import FundusFTExtractor
import numpy as np
import torch
import proj_util
from fundus_v2 import fundus_img_augmentor, fundus_dataloader, fundus_cnn_module, fundus_cnn_trainer, fundus_img_dataset,cnn_model_wrapper, fundus_rf_classifier, fundus_svm_classifier
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

densenet121 = 'densenet121'
resnext50_32x4d = 'resnext50_32x4d'
vgg = 'vgg'
convnext_tiny = 'convnext_tiny'


def run_ml_model_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_models = [
        cnn_model_wrapper.CNNModelWrapper('densenet121'),
        cnn_model_wrapper.CNNModelWrapper('resnext50_32x4d')
    ]
    
    rf_classifier = fundus_rf_classifier.FundusRFClassifier(cnn_models=cnn_models, val_loader=val_loader, device=device)
    # Train and evaluate the classifier
    rf_classifier.train()
    rf_classifier.evaluate()
    
    
    
    # # # # Train and evaluate SVM
    # svm_classifier = fundus_svm_classifier.FundusSVMClassifier(cnn_models=cnn_models, val_loader=val_loader, device=device)
    # # Train and evaluate the classifier
    # svm_classifier.train()
    # svm_classifier.evaluate()
    
    
    
    

def run_cnn_model_training():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_trainer = fundus_cnn_trainer.FundusCNNTrainer(model_name='resnext50_32x4d', train_loader=train_loader, val_loader=val_loader)
    cnn_trainer.train(num_epochs=15)
    val_loss, val_acc, val_metrics = cnn_trainer.validate(val_loader)
    print(f'Validation Loss: {val_loss:.4f} Validation Accuracy: {val_acc:.4f}')
    print(f'Validation Metrics: {cnn_trainer.print_formatted_metrics(val_metrics)}')
    
    cnn_models = [cnn_model_wrapper.CNNModelWrapper('densenet121').model().to(device), 
                  cnn_model_wrapper.CNNModelWrapper('resnext50_32x4d').model().to(device)]
        
    ensemble_model_trainer = fundus_el_trainer.FundusEnsembleModelTrainer(cnn_models, train_loader=train_loader, val_loader=val_loader)
    trained_model = ensemble_model_trainer.train_model()
    
def run_cnn_model_kfold_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = fundus_img_dataset.FundusImageDataset(
                image_dir=proj_util.TRAINING_DIR,
                csv_file=proj_util.TRAIN_LABEL_PATH,
                transform=None,
                train=True,
                num_augmentations=1
            )
    
    # # cnn_trainer.mod_kfold(dataset, num_epochs=10, n_splits=10)
    # # cnn_trainer.grid_search_kfold(dataset, num_epochs=10, n_splits=10)
    # # cnn_trainer.train_kfold(dataset, num_epochs=24, n_splits=15)
    # # cnn_trainer.train_loocv(dataset, num_epochs=5)
    
    
    
    cnn_trainer = fundus_cnn_trainer.FundusCNNTrainer(model_name=densenet121, train_loader=train_loader, val_loader=val_loader)
    cnn_trainer.train(num_epochs=15)
    val_loss, val_acc, val_metrics = cnn_trainer.validate(val_loader)
    print(f'Validation Loss: {val_loss:.4f} Validation Accuracy: {val_acc:.4f}')
    print(f'Validation Metrics: {cnn_trainer.print_formatted_metrics(val_metrics)}')
    
    
    
    cnn_models = [cnn_model_wrapper.CNNModelWrapper('densenet121').model().to(device), 
                  cnn_model_wrapper.CNNModelWrapper('resnext50_32x4d').model().to(device)]
        
    ensemble_model_trainer = fundus_el_trainer.FundusEnsembleModelTrainer(cnn_models, train_loader=train_loader, val_loader=val_loader)
    trained_model = ensemble_model_trainer.train_model()
    

if __name__ == "__main__":
    # run_cnn_model_training()
    run_ml_model_training()