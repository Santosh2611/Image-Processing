import cv2 # Wrapper package for OpenCV python bindings
import numpy as np # General-purpose array-processing package

img = cv2.imread('Lighthouse.jpg') # Read the input image
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert BGR to HSV

# Define range of yellow color in HSV
lower_yellow = np.array([15,50,180])
upper_yellow = np.array([40,255,255])

# Create a mask
rectangular_mask = np.zeros(img.shape[:2], np.uint8)
rectangular_mask[100:250, 150:450] = 255

mask_track = cv2.inRange(hsv, lower_yellow, upper_yellow) # Create a mask; Threshold the HSV image to get only yellow colors
yellow_result = cv2.bitwise_and(img, img, mask= mask_track) # Bitwise-AND mask and original image
rectangular_masked_img = cv2.bitwise_and(img, img, mask = rectangular_mask) # Compute the bitwise AND using the mask

# Save an image to any storage device
cv2.imwrite('Mask Tracking.jpg', mask_track)
cv2.imwrite('Yellow Masked Image.jpg', yellow_result)
cv2.imwrite('Rectangular Mask.jpg', rectangular_mask)
cv2.imwrite('Rectangular Masked Image.jpg', rectangular_masked_img)
