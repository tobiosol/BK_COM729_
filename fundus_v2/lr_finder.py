import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class LRFinder:
    def __init__(self, model, device):
        self.model = model
        LR_FOUND = 2e-3
        
        # Check if the model has 'features' and 'classifier' attributes
        lr_params = []
        if hasattr(model, 'features'):
            lr_params.append({'params': model.features.parameters(), 'lr': LR_FOUND / 10})
        if hasattr(model, 'classifier'):
            lr_params.append({'params': model.classifier.parameters(), 'lr': LR_FOUND})
        else:
            # Handle models without 'classifier' attribute
            lr_params.append({'params': model.parameters(), 'lr': LR_FOUND})
        
        self.optimizer = optim.AdamW(lr_params, lr=1e-7)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.device = device
        self.history = {'lr': [], 'loss': []}

    def range_test(self, train_loader, end_lr=10, num_iter=100):
        def lr_lambda(x):
            return (end_lr / 1e-7) ** (x / num_iter)
        
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.model.train()

        for i, (inputs, targets) in enumerate(train_loader):
            if i >= num_iter:
                break
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            scheduler.step()

            self.history['lr'].append(scheduler.get_last_lr()[0])
            self.history['loss'].append(loss.item())

    def plot(self):
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.show()
