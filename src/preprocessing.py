import cv2

from data_loader import image_loader

def prep(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    equalized = cv2.equalizeHist(blurred) # improves contrast
    return equalized
    