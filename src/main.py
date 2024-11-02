from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
import importlib
import sys
import os

# For uia training 
base_dir = Path(os.getcwd()) / 'solarscan/solarscan/src'
sys.path.append(str(base_dir))

import model
import config

importlib.reload(model)
importlib.reload(config)

#TODO: Do some hyperparameter training with optuna

# Model checkpoints
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss_epoch',
    dirpath=base_dir / 'tmp/checkpoints',
    filename='SOLARSCANMODEL-{epoch:02d}-{val_loss_epoch:.2f}',
    mode='min'
)

# Early stopping
early_stop_callback = pl.callbacks.EarlyStopping(
    monitor='val_loss_epoch',
    patience=5,
    mode='min'
)

# TODO: Add more augmentations because the dataset is small
transform = transforms.Compose([
                transforms.Resize(config.IMG_SIZE),
                transforms.ToTensor()
            ])

image_datasets = {
    'train': datasets.ImageFolder(base_dir / '../dataset/train', transform),
    'val': datasets.ImageFolder(base_dir / '../dataset/val', transform)
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=config.BATCH_SIZE, shuffle=True),
    'val': DataLoader(image_datasets['val'], batch_size=config.BATCH_SIZE, shuffle=False)
}

num_classes = len(config.CLASS_NAMES_2)

model = model.SOLARSCANMODEL(
    num_classes, 
    learning_rate=config.LEARNING_RATE, 
    patience=config.LR_PATIENCE, 
    factor=config.LR_FACTOR
    )

trainer = pl.Trainer(max_epochs=75, callbacks=[checkpoint_callback, early_stop_callback])
trainer.fit(model, dataloaders['train'], dataloaders['val'])