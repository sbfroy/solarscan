from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import pytorch_lightning as pl
import os
import torch

base_dir = Path(os.getcwd())

from model import SOLARSCANMODEL
from config import IMG_SIZE, BATCH_SIZE, LEARNING_RATE, LR_PATIENCE, LR_FACTOR

transform = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor()
            ])

image_datasets = {
    'test': datasets.ImageFolder(base_dir / 'data/images/test', transform)
}

dataloaders = {
    'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False)
}

class_names = image_datasets['test'].classes
num_classes = len(class_names)

model = SOLARSCANMODEL(
    num_classes, 
    learning_rate=LEARNING_RATE,
    patience=LR_PATIENCE,
    factor=LR_FACTOR
    )

model.load_state_dict(torch.load(base_dir / 'src/checkpoints/SOLARSCANMODEL_RESNET50_weights_v6.pth', map_location=torch.device('cpu')))
model.eval()

trainer = pl.Trainer()
trainer.test(model, dataloaders['test'])
