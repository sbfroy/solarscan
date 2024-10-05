import cv2


def image_loader(img_path):
    image = cv2.imread(img_path)
    return image
