import numpy as np
import cv2

from data_loader import single_image_loader
from preprocessing import prep
from config import IMG_SIZE


def detect_edges(preprocessed_image):
    """
    Detects edges in the preprocessed image using Canny edge detection.

    Args:
        preprocessed_image (): 

    Returns: 
        numpy.ndarray: binary image with edges (255) detected.
    """

    median = np.median(preprocessed_image)
    lower = int(max(0, 0.66 * median)) # gradient intensities below lower are discarded
    upper = int(min(255, 1.33 * median)) # values above upper is definitely an edge

    edges = cv2.Canny(preprocessed_image, threshold1=lower, threshold2=upper)

    kernel = np.ones((5, 5), np.uint8) # kernel size = 5x5
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) # closes gaps in between edges

    return closed_edges

def find_contours(edges, image):
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    return contours
    
def isolate_panel(image):
    """
    The pipeline for panel isolation.

    Args: 
        image (): Unprocessed image from data_loader.py

    Returns:
        numpy.ndarray: an adjustable array that reflects how many and the location of panels. 
        along with info about the panel cluster.

    """
    preprocessed_image = prep(image)
    edges = detect_edges(preprocessed_image)
    contours = find_contours(edges, image)

    panel = None
    biggest_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100000:
            continue

        if area > biggest_area:
            biggest_area = area
            x, y, w, h = cv2.boundingRect(contour)
            panel = image[y:y+h, x:x+w]
    
    panel = cv2.resize(panel, IMG_SIZE)

    return panel
    

def main():

    image = single_image_loader("src/../data/images/Dusty/Dust (90).jpg")
    panel = isolate_panel(image)

    cv2.imshow("", panel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        

if __name__ == "__main__":
    main()