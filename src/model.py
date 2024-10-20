import torch
import torch.nn as nn
from torchvision import models
import pytorch_lightning as pl
import torch.nn.functional as F

from config import LEARING_RATE


class SOLARSCANMODEL(pl.LightningModule):
    def __init__(self, num_classes):
        super(SOLARSCANMODEL, self).__init__()
        self.model = models.resnet50(pretrained=True)
        # Replace the final layer with our number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes) 

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean()
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARING_RATE)
        return optimizer
