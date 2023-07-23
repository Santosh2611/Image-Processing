import cv2  # Wrapper package for OpenCV python bindings
import numpy as np  # General-purpose array-processing package
from matplotlib import pyplot as plt # Collection of command style functions that make matplotlib work like MATLAB

# Load the input image in grayscale
img = cv2.imread('Lighthouse.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate the histogram using the OpenCV function cv2.calcHist()
hist, bins = np.histogram(img.flatten(), 256, [0, 256])

# Plot the histogram using Matplotlib
plt.hist(img.flatten(), 256, [0, 256])
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Grayscale Image Histogram')

# Show the histogram
plt.show()
