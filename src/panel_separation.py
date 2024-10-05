import cv2

from data_loader import image_loader
from preprocessing import prep


def detect_edges(image, threshold1, threshold2):
    edges = cv2.Canny(image, threshold1=threshold1, threshold2=threshold2)
    return edges

def find_contours(image, original_image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    contours_image = cv2.drawContours(original_image, contours, -1, (0, 255, 0), 2)
    return contours_image
    

def main():
    image = image_loader("src/../data/images/IMG_0492.jpg")
    preprocessed_image = prep(image)
    edges = detect_edges(preprocessed_image, threshold1=75, threshold2=150)
    contours = find_contours(edges, image)
  
    cv2.imshow("Contours", contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()