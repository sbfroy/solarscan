import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import pytorch_lightning as pl
import os

base_dir = Path(os.getcwd())

from model import SOLARSCANMODEL
from config import *

transform = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor()
            ])

image_datasets = {
    'test': datasets.ImageFolder(base_dir / 'dataset/test', transform)
}

dataloaders = {
    'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False)
}

num_classes = len(CLASS_NAMES)
model = SOLARSCANMODEL(
    num_classes=num_classes, 
    learning_rate=LEARNING_RATE,
    patience=LR_PATIENCE,
    factor=LR_FACTOR
)

model.load_state_dict(torch.load(base_dir / 'src/checkpoints/SOLARSCANMODEL_weights_RESNET50_BINARY.pth', map_location=torch.device('cpu')))
model.eval()

trainer = pl.Trainer()
trainer.test(model, dataloaders['test'])
