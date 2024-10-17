import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import cv2
import os
from pathlib import Path

from config import IMG_SIZE
from panel_separation import isolate_panels







def model():
    # test different pre-trained models, like YOLOv8, ImageNet, etc.
    pass


# Important to resize all the images ti IMG_SIZE


def main():

    base_dir = os.path.dirname(__file__)
    img_path = Path(base_dir).parent / "data/images/Dusty/dust (37).jpg"

    image = isolate_panels(img_path)

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
