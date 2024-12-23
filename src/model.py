import torch
import torch.nn as nn
from torchvision import models
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics

class SOLARSCANMODEL(pl.LightningModule):
    def __init__(self, num_classes, learning_rate, patience, factor):
        super(SOLARSCANMODEL, self).__init__()
        self.save_hyperparameters()
        self.model = models.resnet18(pretrained=True)

        self.model.fc = nn.Sequential(
            nn.Dropout(0.3), 
            nn.Linear(self.model.fc.in_features, num_classes)
        )
        
        self.learning_rate = learning_rate
        self.confusion_matrix = torchmetrics.ConfusionMatrix('binary')

        # For logging purposes
        self.t_outputs = []
        self.v_outputs = []

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) # Predictions
        loss = F.cross_entropy(y_hat, y)
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean()
        self.t_outputs.append({'loss': loss, 'acc': acc})
        return {'loss': loss, 'acc': acc}
    
    def on_train_epoch_end(self):
        avg_t_loss = torch.stack([x['loss'] for x in self.t_outputs]).mean()
        avg_t_acc = torch.stack([x['acc'] for x in self.t_outputs]).mean()

        avg_v_loss = torch.stack([x['loss'] for x in self.v_outputs]).mean()
        avg_v_acc = torch.stack([x['acc'] for x in self.v_outputs]).mean()

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        
        self.log_dict({
            'train_loss_epoch': avg_t_loss,
            'train_acc_epoch': avg_t_acc,
            'val_loss_epoch': avg_v_loss,
            'val_acc_epoch': avg_v_acc,
            'learning_rate': lr
        }, prog_bar=True)

        self.t_outputs.clear()
        self.v_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean()
        self.v_outputs.append({'loss': loss, 'acc': acc})
        return {'loss': loss, 'acc': acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()

        self.confusion_matrix(preds, y)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)    

    def on_test_epoch_end(self):
        conf_matrix = self.confusion_matrix.compute().cpu().numpy()
        print(f'Confusion matrix:\n', conf_matrix)
        self.confusion_matrix.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            'min', 
            patience=self.hparams.patience, 
            factor=self.hparams.factor)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss_epoch'
        }
