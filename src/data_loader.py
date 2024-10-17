import cv2
import os

from config import IMG_SIZE

def resize(image):
    return cv2.resize(image, IMG_SIZE)


def single_image_loader(img_path):
    image = cv2.imread(img_path)
    resized = resize(image)
    return resized


def multiple_image_loader(folder_path):
    images = []
    formats = (".jpg", ".jpeg", ".png")
    for img in os.listdir(folder_path):
        if img.endswith(formats):
            image = cv2.imread(os.path.join(folder_path, img))
            resized = resize(image)
            images.append(resized)
    return images

def main():
    multiple_image_loader(folder_path="C:/Users/Philip Haugen/solarscan/data/images/Clean")
    print(len(multiple_image_loader(folder_path="C:/Users/Philip Haugen/solarscan/data/images/Clean")))

if __name__ == "__main__":
    main()