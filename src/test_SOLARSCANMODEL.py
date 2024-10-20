from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import pytorch_lightning as pl
import os
import torch

base_dir = Path(os.getcwd())

from model import SOLARSCANMODEL
from config import IMG_SIZE

data_transforms = {
    'test': transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor()
            ])      
    }

image_datasets = {
    'test': datasets.ImageFolder(base_dir / 'data/images/test', data_transforms['test'])
}

dataloaders = {
    'test': DataLoader(image_datasets['test'], batch_size=4, shuffle=False)
}

class_names = image_datasets['test'].classes
num_classes = len(class_names)

model = SOLARSCANMODEL(num_classes)

model.load_state_dict(torch.load(base_dir / 'src/SOLARSCANMODEL_RESNET50_weights.pth', map_location=torch.device('cpu')))
model.eval()

trainer = pl.Trainer()
trainer.test(model, dataloaders['test'])
