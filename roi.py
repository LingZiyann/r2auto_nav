import numpy as np
import cv2 as cv

class ROI:
    # Initialize class attributes
    area = 0  # To store the area of the ROI
    vertices = None  # To store the coordinates of the ROI's vertices

    def init_roi(self, width, height):
        # Define the vertices of the ROI based on the image dimensions.
        # The vertices define a trapezoid shape at the bottom of the image.
        vertices = [(0, height), (width / 4, 3 * height / 4),
                    (3 * width / 4, 3 * height / 4), (width, height),]
        self.vertices = np.array([vertices], np.int32)
        
        # Create a blank image (all white) with the same dimensions as the input image.
        blank = np.zeros((height, width, 3), np.uint8)
        blank[:] = (255, 255, 255)  # Fill the blank image with white color
        blank_gray = cv.cvtColor(blank, cv.COLOR_BGR2GRAY)  # Convert the blank image to grayscale
        
        # Crop the blank image using the defined ROI to calculate the ROI's area.
        blank_cropped = self.crop_roi(blank_gray)
        self.area = cv.countNonZero(blank_cropped)  # Count the non-zero (white) pixels in the cropped area

    def crop_roi(self, img):
        # Create a mask with the same size as the input image, initially all zeros (black).
        mask = np.zeros_like(img)
        match_mask_color = 255  # Define the mask color (white) to fill the ROI
        
        # Fill the defined ROI polygon in the mask with white color.
        cv.fillPoly(mask, self.vertices, match_mask_color)
        # Apply the mask to the input image, keeping only the ROI area.
        masked_image = cv.bitwise_and(img, mask)
        return masked_image  # Return the image cropped to the ROI

    def get_area(self):
        # Return the pre-calculated area of the ROI
        return self.area

    def get_vertices(self):
        # Return the vertices defining the ROI
        return self.vertices
