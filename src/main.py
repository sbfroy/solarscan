from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import pytorch_lightning as pl
import torch
import sys
import os

# For uia training 
base_dir = Path(os.getcwd()) / 'solarscan/solarscan/src'
sys.path.append(str(base_dir))

from model import SOLARSCANMODEL
from config import IMG_SIZE

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    dirpath=base_dir / 'checkpoints',
    filename='solarPanelClassifier-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min'
)

data_transforms = {
    'train': transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor()
            ]),
    'val': transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor()
            ])     
    }

image_datasets = {
    'train': datasets.ImageFolder(base_dir / '../data/images/train', data_transforms['train']),
    'val': datasets.ImageFolder(base_dir / '../data/images/val', data_transforms['val'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=4, shuffle=True),
    'val': DataLoader(image_datasets['val'], batch_size=4, shuffle=False)
}

class_names = image_datasets['train'].classes
num_classes = len(class_names)

model = SOLARSCANMODEL(num_classes)

trainer = pl.Trainer(max_epochs=10, callbacks=[checkpoint_callback])
trainer.fit(model, dataloaders['train'], dataloaders['val'])
