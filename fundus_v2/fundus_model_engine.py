import os
import sys
from pathlib import Path

import torch



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))
subdirectory_path = Path(current_dir) / 'fundus_v2'
sys.path.append(str(subdirectory_path))

import proj_util

# Import necessary modules
from fundus_v2 import (
    fundus_el_model,
    fundus_el_trainer,
    fundus_ftextractor,
    fundus_img_augmentor,
    fundus_dataloader,
    fundus_cnn_module,
    fundus_cnn_trainer,
    fundus_img_dataset,
    cnn_model_wrapper,
    fundus_rf_classifier,
    fundus_svm_classifier
)

# Set up device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FundusModelEngine:
    
    def __init__(self):
        # Initialize augmentor and transforms
        self.augmentor = fundus_img_augmentor.FundusImageAugmentor()
        self.train_transform = self.augmentor.train_transform
        self.test_transform = self.augmentor.test_transform
        
        # Define CNN models
        self.densenet121 = 'densenet121'
        self.resnext50_32x4d = 'resnext50_32x4d'
        self.vgg = 'vgg'
        self.convnext_tiny = 'convnext_tiny'

        # Initialize data loader
        self.dataloader = fundus_dataloader.FundusDataLoader(image_dir=proj_util.TRAINING_DIR, csv_file=proj_util.TRAIN_LABEL_PATH,
                                                                batch_size=32, transform=self.train_transform, train=True, num_augmentations=5)
        self.train_loader = self.dataloader.get_loader()

        self.dataloader = fundus_dataloader.FundusDataLoader(image_dir=proj_util.VALIDATION_DIR, csv_file=proj_util.VALIDATION_LABEL_PATH,
                                                                batch_size=32, transform=self.test_transform, shuffle=False, train=False, num_augmentations=1)
        self.val_loader = self.dataloader.get_loader()

        # Initialize models and data loaders
        self.cnn_models = [
            cnn_model_wrapper.CNNModelWrapper('densenet121'),
            cnn_model_wrapper.CNNModelWrapper('resnext50_32x4d')
        ]


    def train_cnn_models(self):
        """Train CNN models"""
        cnn_trainer = fundus_cnn_trainer.FundusCNNTrainer(model_name=self.densenet121, train_loader=self.train_loader, val_loader=self.val_loader)
        cnn_trainer.train(num_epochs=15)
        val_loss, val_acc, val_metrics = cnn_trainer.validate(self.val_loader)
        print(f'Validation Loss: {val_loss:.4f} Validation Accuracy: {val_acc:.4f}')
        print(f'Validation Metrics: {cnn_trainer.print_formatted_metrics(val_metrics)}')
        
        
        # ensemble_model_trainer = fundus_el_trainer.FundusEnsembleModelTrainer(self.cnn_models, train_loader=self.train_loader, val_loader=self.val_loader)
        # trained_model = ensemble_model_trainer.train_model()

    def train_rf_classifier(self):
        """Train random forest classifier"""
        self.rf_classifier = fundus_rf_classifier.FundusRFClassifier(cnn_models=self.cnn_models, val_loader=self.val_loader, device=device)
        self.rf_classifier.train()
        self.rf_classifier.evaluate()
        

    def train_svm_classifier(self):
        """Train support vector machine classifier"""
        self.svm_classifier = fundus_svm_classifier.FundusSVMClassifier(cnn_models=self.cnn_models, val_loader=self.val_loader, device=device)
        self.svm_classifier.train()
        self.svm_classifier.evaluate()
        
    def train_svm_el_classifier(self):
        """Train support vector machine classifier"""
        self.svm_classifier = fundus_svm_classifier.FundusSVMClassifier(cnn_models=self.cnn_models, val_loader=self.val_loader, device=device)
        self.svm_classifier.train_el()
        # self.svm_classifier.predict_el()
        self.svm_classifier.evaluate_el()


if __name__ == "__main__":
    # Run the model training functions
    trainer = FundusModelEngine()
    # trainer.train_cnn_models()
    # trainer.train_rf_classifier()
    trainer.train_svm_el_classifier()