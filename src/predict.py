import torch
from torchvision import transforms
import os
from pathlib import Path
import cv2
from PIL import Image

from data_loader import single_image_loader
from panel_separation import isolate_panels

from model import SOLARSCANMODEL
from config import *

base_dir = os.path.dirname(__file__)

transform = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor()
            ])

num_classes = len(CLASS_NAMES)
model = SOLARSCANMODEL(
    num_classes=num_classes, 
    learning_rate=LEARNING_RATE,
    patience=LR_PATIENCE,
    factor=LR_FACTOR
)

model.load_state_dict(torch.load(Path(base_dir) / 'checkpoints/SOLARSCANMODEL_weights_RESNET18_v2.pth', map_location=torch.device('cpu')))
model.eval()

for image in os.listdir(Path(base_dir).parent / "dataset/test/good"):
    img_path = Path(base_dir).parent / "dataset/test/good" / image
    unprocessed_image = single_image_loader(img_path)
    panel = isolate_panels(unprocessed_image)

    cv2.imshow("unprocessed_image", unprocessed_image)
    cv2.imshow("panel", panel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # If i make pred on panel it is bad, but good on whole image. 
    # maybe it does not look enough at the panel to make its predictions
    image = Image.fromarray(unprocessed_image) # Ensure it is a PIL image
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        print("Model output:", output)
        _, predicted = torch.max(output, 1)

    predicted = CLASS_NAMES[predicted.item()]

    print(f'I think the panel is {predicted}.')