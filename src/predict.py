import torch
from torchvision import transforms
import os
from pathlib import Path
import cv2
from PIL import Image

from data_loader import single_image_loader
from preprocessing import prep
from panel_separation import isolate_panels

from model import SOLARSCANMODEL
from config import IMG_SIZE, LEARNING_RATE, LR_PATIENCE, LR_FACTOR, CLASS_NAMES, CLASS_NAMES_2

base_dir = os.path.dirname(__file__)

transform = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor()
            ])

num_classes = len(CLASS_NAMES_2)
model = SOLARSCANMODEL(
    num_classes=num_classes, 
    learning_rate=LEARNING_RATE,
    patience=LR_PATIENCE,
    factor=LR_FACTOR
)

model.load_state_dict(torch.load(Path(base_dir) / 'checkpoints/SOLARSCANMODEL_RESNET50_weights_v6.pth', map_location=torch.device('cpu')))
model.eval()

"data/images/val/Clean/Clean (17).jpg"

"data/images/train/Dusty/Dust (130).jpg"

"data/images/train/Dusty/Dust (153).jpg"

# "data/images/test/Clean/Clean (127).jpg" FAILS
# "data/images/test/Clean/Clean (184).jpg" FAILS (ISOLATE A FRACTION OF PANEL)

img_path = Path(base_dir).parent / "data/images/test/Bird-drop/Bird (160).jpg"
image = single_image_loader(img_path)
panel = isolate_panels(image)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("panel", panel)
cv2.waitKey(0)
cv2.destroyAllWindows()

image = Image.fromarray(panel) # Ensure it is a PIL image
image = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

predicted = CLASS_NAMES[predicted.item()]

print(f'I think: {predicted}')
