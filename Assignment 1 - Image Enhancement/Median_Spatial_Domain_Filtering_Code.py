import cv2 # Wrapper package for OpenCV python bindings
import numpy as np # General-purpose array-processing package

img_noisy = cv2.imread('Lighthouse.jpg', 0) # Read the input image
m, n = img_noisy.shape # Obtain the number of rows and columns of the image
img_new = np.zeros([m, n]) # Initialize a new image with zeros

# Traverse the image and apply the median filter to each 3x3 neighborhood
for i in range(1, m-1):
	for j in range(1, n-1):
        
        # Get the nine pixels in the 3x3 neighborhood
		temp = [img_noisy[i-1, j-1],
			img_noisy[i-1, j],
			img_noisy[i-1, j + 1],
			img_noisy[i, j-1],
			img_noisy[i, j],
			img_noisy[i, j + 1],
			img_noisy[i + 1, j-1],
			img_noisy[i + 1, j],
			img_noisy[i + 1, j + 1]]
		
		temp = sorted(temp) # Returns a sorted list of the specified iterable object
		img_new[i, j]= temp[4] # Select the median value

img_new = img_new.astype(np.uint8) # Convert the data type of the output image to uint8
cv2.imwrite('Median Spatial Domain Filtering - Lighthouse.jpg', img_new) # Save an image to any storage device
