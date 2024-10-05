import cv2

IMG_SIZE = (768, 768)

def image_loader(img_path):
    image = cv2.imread(img_path)
    resized_image = cv2.resize(image, IMG_SIZE)
    return resized_image
