import cv2
import numpy as np
from tqdm import tqdm

from preprocessing import prep

hsv_range = {
    'h': (70, 160),       
    's': (0, 90)       
}

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
    
def color_check(image, contour, threshold):
    """
    Checks the standard deviation of the color in the contour. If the standard deviation is below the threshold, 
    the area is considered uniform in color.
        
    Returns:
        bool: True if the standard deviation is below the threshold, False otherwise.
    """
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1) 
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    pixels = masked_image[mask == 255]
    _, stddev = cv2.meanStdDev(pixels)
    return np.all(stddev < threshold)

def color_range_check(image, contour, hsv_range):
    """
    Checks if the contours mean hsv value is within hsv_range (panel range).
    
    Returns:
        bool: True if true, False otherwise.
    """

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)  # Fill the contour area

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    masked_image = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)

    mean_hsv = np.mean(masked_image[mask == 255], axis=0)

    #print("Mean HSV: ", mean_hsv)

    hue_in_range = hsv_range['h'][0] <= mean_hsv[0] <= hsv_range['h'][1]
    sat_in_range = hsv_range['s'][0] <= mean_hsv[1] <= hsv_range['s'][1]
    
    return hue_in_range and sat_in_range  

def isolate_panels(image):
    """
    The whole pipeline for panel isolation.

    Args: 
        image (): Unprocessed image from data_loader.py

    Returns:
        numpy.ndarray: Returns the "best" image that passes the color_range_check.
    """
    preprocessed_image = prep(image)
    edges = detect_edges(preprocessed_image)
    contours = find_contours(edges, image)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in tqdm(sorted_contours[1:]):
        x, y, w, h = cv2.boundingRect(contour)

        if color_range_check(image, contour, hsv_range) and color_check(image, contour, 57):
            panel = image[y:y+h, x:x+w]
            return panel

    return None
