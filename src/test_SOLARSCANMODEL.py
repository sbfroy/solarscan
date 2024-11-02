from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import pytorch_lightning as pl
import os
import torch

base_dir = Path(os.getcwd())

from model import SOLARSCANMODEL
from config import IMG_SIZE, BATCH_SIZE, LEARNING_RATE, LR_PATIENCE, LR_FACTOR, CLASS_NAMES_2

transform = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor()
            ])

image_datasets = {
    'test': datasets.ImageFolder(base_dir / 'data_4/test', transform)
}

dataloaders = {
    'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False)
}

model = SOLARSCANMODEL(
    num_classes=len(CLASS_NAMES_2), 
    learning_rate=LEARNING_RATE,
    patience=LR_PATIENCE,
    factor=LR_FACTOR
    )

model.load_state_dict(torch.load(base_dir / 'src/checkpoints/SOLARSCANMODEL_weights_2classes.pth', map_location=torch.device('cpu')))
model.eval()

trainer = pl.Trainer()
trainer.test(model, dataloaders['test'])
