import cv2 # Wrapper package for OpenCV python bindings
import numpy as np # General-purpose array-processing package
import warnings; warnings.filterwarnings("ignore")

img = cv2.imread('Lighthouse.jpg') # Read the input image
  
# Log Transformation
c = 255/(np.log(1 + np.max(img))) # Calculate the constant c
log_transformed = c * np.log(1 + img) # Apply the log transform  
log_transformed = np.array(log_transformed, dtype = np.uint8) # Convert the data type to unit8
cv2.imwrite('Log Transformed - Lighthouse.jpg', log_transformed) # Save the log-transformed image

# Power-Law (Gamma) Transformations
for gamma in [0.1, 0.5, 1.2]: # Iterate over different gamma values
    gamma_corrected = np.array(255*(img / 255) ** gamma, dtype = 'uint8') # Apply gamma correction
    cv2.imwrite('Gamma Transformed ('+str(gamma)+').jpg', gamma_corrected) # Save the gamma corrected image
