import importlib

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

import model
import config

importlib.reload(model)
importlib.reload(config)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss_epoch',
    dirpath=base_dir / 'tmp/checkpoints',
    filename='SOLARSCANMODEL-{epoch:02d}-{val_loss_epoch:.2f}',
    save_top_k=3,
    mode='min'
)

early_stop_callback = pl.callbacks.EarlyStopping(
    monitor='val_loss_epoch',
    patience=5,
    mode='min'
)

transform = transforms.Compose([
                transforms.Resize(config.IMG_SIZE),
                transforms.ToTensor()
            ])

image_datasets = {
    'train': datasets.ImageFolder(base_dir / '../data/images/train', transform),
    'val': datasets.ImageFolder(base_dir / '../data/images/val', transform)
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=config.BATCH_SIZE, shuffle=True),
    'val': DataLoader(image_datasets['val'], batch_size=config.BATCH_SIZE, shuffle=False)
}

class_names = image_datasets['train'].classes
num_classes = len(class_names)

model = model.SOLARSCANMODEL(
    num_classes, 
    learning_rate=config.LEARNING_RATE, 
    patience=config.LR_PATIENCE, 
    factor=config.LR_FACTOR
    )

trainer = pl.Trainer(max_epochs=5, callbacks=[checkpoint_callback, early_stop_callback])
trainer.fit(model, dataloaders['train'], dataloaders['val'])