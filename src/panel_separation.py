import numpy as np
import cv2

from data_loader import single_image_loader
from preprocessing import prep


def detect_edges(image):
    # gradient intensities below threshold1 are discarded
    # values above threshold 2 is definitely an edge
    median = np.median(image)
    lower = int(max(0, 0.33 * median))
    upper = int(min(255, 1.6 * median))
    edges = cv2.Canny(image, threshold1=lower, threshold2=upper)
    return edges

def enhance_edges(edges):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    return dilated

def find_contours(edges, image):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    return image
    

def main():
    image = single_image_loader("C:/Users/Philip Haugen/solarscan/data/images/Clean/Clean (30).jpg")

    preprocessed_image = prep(image)
    edges = detect_edges(preprocessed_image)
    enhanced_edges = enhance_edges(edges)
    contours = find_contours(enhanced_edges, image)
  
    cv2.imshow("Contours", contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()