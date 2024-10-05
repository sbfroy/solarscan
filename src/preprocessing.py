import cv2

from data_loader import image_loader

IMG_SIZE = (768, 768)

def prep(image):
    resized_image = cv2.resize(image, IMG_SIZE)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY) 
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    return blurred_image
    