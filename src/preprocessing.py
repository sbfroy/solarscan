import cv2

def prep(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return blurred
    