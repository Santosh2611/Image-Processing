import cv2 # Wrapper package for OpenCV python bindings
import numpy as np # General-purpose array-processing package

img = cv2.imread('Lighthouse.jpg', 0) # Read the input image
m,n = img.shape # To ascertain total numbers of rows and columns of the image, size of the image
L = img.max() # To find the maximum grey level value in the image
img_neg = L-img # Maximum grey level value minus the original image gives the negative image
cv2.imwrite('Image Negative - Lighthouse.png', img_neg) # Convert the np array img_neg to a png image

# Thresholding without background
# Let threshold =T
# Let pixel value in the original be denoted by r
# Let pixel value in the new image be denoted by s
# If r<T, s= 0
# If r>T, s=255

T = 150
img_thresh = np.zeros((m,n), dtype = int) # Create an array of zeros

for i in range(m):	
	for j in range(n):
		
		if img[i,j] < T:
			img_thresh[i,j]= 0
		else:
			img_thresh[i,j] = 255

cv2.imwrite('Image with Thresholding.png', img_thresh) # Convert array to png image

T1 = 100 # Lower threshold value
T2 = 180 # Upper threshold value
img_thresh_back = np.zeros((m,n), dtype = int) # Create an array of zeros

for i in range(m):	
	for j in range(n):
		
		if T1 < img[i,j] < T2:
			img_thresh_back[i,j]= 255
		else:
			img_thresh_back[i,j] = img[i,j]

cv2.imwrite('Image with Grey Level Slicing with Background.png', img_thresh_back) # Convert array to png image
