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

pl.seed_everything(config.SEED)

# Model checkpoints
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss_epoch',
    dirpath=base_dir / 'tmp/checkpoints',
    filename='SOLARSCANMODEL-{epoch:02d}-{val_loss_epoch:.2f}',
    mode='min',
    save_top_k=5
)

# Early stopping
early_stop_callback = pl.callbacks.EarlyStopping(
    monitor='val_loss_epoch',
    patience=3,
    mode='min'
)

train_transforms = transforms.Compose([
    transforms.Resize(config.IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor()
])

val_test_transforms = transforms.Compose([
                transforms.Resize(config.IMG_SIZE),
                transforms.ToTensor()
            ])

image_datasets = {
    'train': datasets.ImageFolder(base_dir / '../dataset/train', train_transforms),
    'val': datasets.ImageFolder(base_dir / '../dataset/val', val_test_transforms),
    'test': datasets.ImageFolder(base_dir / 'dataset/test', val_test_transforms)
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=config.BATCH_SIZE, shuffle=True),
    'val': DataLoader(image_datasets['val'], batch_size=config.BATCH_SIZE, shuffle=False),
    'test': DataLoader(image_datasets['test'], batch_size=config.BATCH_SIZE, shuffle=False)
}

num_classes = len(config.CLASS_NAMES)

model = model.SOLARSCANMODEL(
    num_classes=num_classes, 
    learning_rate=config.LEARNING_RATE, 
    patience=config.LR_PATIENCE, 
    factor=config.LR_FACTOR
    )

trainer = pl.Trainer(max_epochs=25, callbacks=[checkpoint_callback, early_stop_callback])
trainer.fit(model, dataloaders['train'], dataloaders['val'])
trainer.test(model, dataloaders['test'])
