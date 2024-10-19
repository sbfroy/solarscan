from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import pytorch_lightning as pl
import os

from model import solarPanelClassifier
from config import IMG_SIZE

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints',
    filename='solarPanelClassifier-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min'
)

data_transforms = {
    'train': transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2,        
                                       contrast=0.2,
                                       saturation=0.2,
                                       hue=0.1),
                transforms.ToTensor()
            ]),
    'val': transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor()
            ]),
    'test': transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor()
            ])      
    }

base_dir = os.path.dirname(__file__)

image_datasets = {
    'train': datasets.ImageFolder(Path(base_dir).parent / 'data/images/train', data_transforms['train']),
    'val': datasets.ImageFolder(Path(base_dir).parent / 'data/images/val', data_transforms['test']),
    'test': datasets.ImageFolder(Path(base_dir).parent / 'data/images/test', data_transforms['test'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=4, shuffle=True),
    'val': DataLoader(image_datasets['val'], batch_size=4, shuffle=False),
    'test': DataLoader(image_datasets['test'], batch_size=4, shuffle=False)
}

class_names = image_datasets['train'].classes
num_classes = len(class_names)

model = solarPanelClassifier(num_classes)

trainer = pl.Trainer(max_epochs=10, callbacks=[checkpoint_callback])
trainer.fit(model, dataloaders['train'], dataloaders['val'])
trainer.test(model, dataloaders['test'])
