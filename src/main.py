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
from config import IMG_SIZE, BATCH_SIZE, LEARNING_RATE


checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss_epoch',
    dirpath=base_dir / 'checkpoints',
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
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor()
            ])

image_datasets = {
    'train': datasets.ImageFolder(base_dir / '../data/images/train', transform),
    'val': datasets.ImageFolder(base_dir / '../data/images/val', transform)
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True),
    'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False)
}

class_names = image_datasets['train'].classes
num_classes = len(class_names)

model = SOLARSCANMODEL(num_classes, learning_rate=LEARNING_RATE)

trainer = pl.Trainer(max_epochs=35, callbacks=[checkpoint_callback, early_stop_callback])
trainer.fit(model, dataloaders['train'], dataloaders['val'])
