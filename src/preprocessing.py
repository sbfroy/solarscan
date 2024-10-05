import cv2

from data_loader import image_loader

def prep(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    return blurred_image
    