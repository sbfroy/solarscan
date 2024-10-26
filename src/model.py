import torch
import torch.nn as nn
from torchvision import models
import pytorch_lightning as pl
import torch.nn.functional as F

# TODO: Add other parameters to the logs. and then run a big run on different parameteres automatically.

class SOLARSCANMODEL(pl.LightningModule):
    def __init__(self, num_classes, learning_rate):
        super(SOLARSCANMODEL, self).__init__()
        self.save_hyperparameters()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes) 
        self.learning_rate = learning_rate
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean()
        self.training_step_outputs.append({'loss': loss, 'acc': acc})
        return {'loss': loss, 'acc': acc}
    
    def on_train_epoch_end(self):
        avg_train_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        avg_train_acc = torch.stack([x['acc'] for x in self.training_step_outputs]).mean()

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        
        avg_val_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        avg_val_acc = torch.stack([x['acc'] for x in self.validation_step_outputs]).mean()
        
        self.log_dict({
            'train_loss_epoch': avg_train_loss,
            'train_acc_epoch': avg_train_acc,
            'val_loss_epoch': avg_val_loss,
            'val_acc_epoch': avg_val_acc,
            'learning_rate': lr
        }, prog_bar=True)

        self.training_step_outputs.clear()
        self.validation_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean()
        self.validation_step_outputs.append({'loss': loss, 'acc': acc})
        return {'loss': loss, 'acc': acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean()
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss_epoch'
        }
