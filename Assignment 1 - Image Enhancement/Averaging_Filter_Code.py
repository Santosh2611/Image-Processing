# Low Pass Spatial Domain Filtering to observe the blurring effect
  
import cv2 # Wrapper package for OpenCV python bindings
import numpy as np # General-purpose array-processing package
      
img = cv2.imread('Lighthouse.jpg', 0) # Read the image
m, n = img.shape # Obtain the number of rows and columns of the image
img_new = np.zeros([m, n]) # Convolve the 3X3 mask over the image 
   
# Develop Averaging filter (3, 3) mask
mask = np.ones([3, 3], dtype = int)
mask = mask/9
  
for i in range(1, m-1):    
    for j in range(1, n-1):
        
        temp = img[i-1, j-1]*mask[0, 0] + img[i-1, j]*mask[0, 1] + img[i-1, j + 1]*mask[0, 2] + img[i, j-1]*mask[1, 0] + img[i, j]*mask[1, 1] + img[i, j + 1]*mask[1, 2] + img[i + 1, j-1]*mask[2, 0] + img[i + 1, j]*mask[2, 1] + img[i + 1, j + 1]*mask[2, 2]
         
        img_new[i, j]= temp
          
img_new = img_new.astype(np.uint8) # Convert the data type of the output image to uint8
cv2.imwrite('Averaging Filter - Lighthouse.jpg', img_new) # Save an image to any storage device
