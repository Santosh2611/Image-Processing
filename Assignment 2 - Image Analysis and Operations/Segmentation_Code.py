# Importing necessary Libraries
from skimage import data, filters
from skimage.color import rgb2gray, rgb2hsv, label2rgb
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian
from skimage.segmentation import active_contour, chan_vese, slic, mark_boundaries, felzenszwalb
from skimage.data import astronaut

# Setting the plot size to 15,15
plt.figure(figsize=(15, 15))

# Sample Image of scikit-image package
coffee = data.coffee()
plt.subplot(1, 2, 1)

# Displaying the sample image
plt.imshow(coffee)
plt.title('Original RGB Image')
plt.xlabel('X Label')
plt.ylabel('Y Label')

# Converting RGB image to Monochrome
gray_coffee = rgb2gray(coffee)
plt.subplot(1, 2, 2)

# Displaying the sample image - Monochrome Format
plt.imshow(gray_coffee, cmap="gray")
plt.title('Monochrome Image')
plt.xlabel('X Label')
plt.ylabel('Y Label')

# Setting the plot size to 15,15
plt.figure(figsize=(15, 15))

# Sample Image of scikit-image package
coffee = data.coffee()
plt.subplot(1, 2, 1)

# Displaying the sample image
plt.imshow(coffee)
plt.title('Original RGB Image')
plt.xlabel('X Label')
plt.ylabel('Y Label')

# Converting RGB Image to HSV Image
hsv_coffee = rgb2hsv(coffee)
plt.subplot(1, 2, 2)

# Displaying the sample image - HSV Format
hsv_coffee_colorbar = plt.imshow(hsv_coffee)
plt.title('HSV Image')
plt.xlabel('X Label')
plt.ylabel('Y Label')

# Adjusting colorbar to fit the size of the image
plt.colorbar(hsv_coffee_colorbar, fraction=0.046, pad=0.04)

# Displaying the sample image - Monochrome Format
# Sample Image of scikit-image package
coffee = data.coffee()
gray_coffee = rgb2gray(coffee)

# Setting the plot size to 15,15
plt.figure(figsize=(15, 15))

for i in range(10):
    
    # Iterating different thresholds
    binarized_gray = (gray_coffee > i*0.1)*1
    plt.subplot(5,2,i+1)
    
    # Rounding of the threshold value to 1 decimal point
    plt.title("Threshold: >"+str(round(i*0.1,1)))
    plt.xlabel('X Label')
    plt.ylabel('Y Label')
    
    # Displaying the binarized image
    # of various thresholds
    plt.imshow(binarized_gray, cmap = 'gray')

plt.tight_layout()

# Setting plot size to 15, 15
plt.figure(figsize=(15, 15))

# Sample Image of scikit-image package
coffee = data.coffee()
gray_coffee = rgb2gray(coffee)

# Computing Otsu's thresholding value
threshold = filters.threshold_otsu(gray_coffee)

# Computing binarized values using the obtained threshold
binarized_coffee = (gray_coffee > threshold)*1
plt.subplot(2,2,1)
plt.title("Threshold: >"+str(threshold))
plt.xlabel('X Label')
plt.ylabel('Y Label')

# Displaying the binarized image
plt.imshow(binarized_coffee, cmap = "gray")

# Computing Ni black's local pixel threshold values for every pixel
threshold = filters.threshold_niblack(gray_coffee)

# Computing binarized values using the obtained threshold
binarized_coffee = (gray_coffee > threshold)*1
plt.subplot(2,2,2)
plt.title("Niblack Thresholding")
plt.xlabel('X Label')
plt.ylabel('Y Label')

# Displaying the binarized image
plt.imshow(binarized_coffee, cmap = "gray")

# Computing Sauvola's local pixel threshold values for every pixel - Not Binarized
threshold = filters.threshold_sauvola(gray_coffee)
plt.subplot(2,2,3)
plt.title("Sauvola Thresholding")
plt.xlabel('X Label')
plt.ylabel('Y Label')

# Displaying the local threshold values
plt.imshow(threshold, cmap = "gray")

# Computing Sauvola's local pixel threshold values for every pixel - Binarized
binarized_coffee = (gray_coffee > threshold)*1
plt.subplot(2,2,4)
plt.title("Sauvola Thresholding - Converting to 0's and 1's")
plt.xlabel('X Label')
plt.ylabel('Y Label')

# Displaying the binarized image
plt.imshow(binarized_coffee, cmap = "gray")

# Sample Image of scikit-image package
astronaut = astronaut()
gray_astronaut = rgb2gray(astronaut)

# Applying Gaussian Filter to remove noise
gray_astronaut_noiseless = gaussian(gray_astronaut, 1)

# Localising the circle's center at 220, 110
x1 = 220 + 100*np.cos(np.linspace(0, 2*np.pi, 500))
x2 = 100 + 100*np.sin(np.linspace(0, 2*np.pi, 500))

# Generating a circle based on x1, x2
snake = np.array([x1, x2]).T

# Computing the Active Contour for the given image
astronaut_snake = active_contour(gray_astronaut_noiseless,
                                snake)

fig = plt.figure(figsize=(10, 10))

# Adding subplots to display the markers
ax = fig.add_subplot(111)

# Plotting sample image
ax.imshow(gray_astronaut_noiseless)
plt.title('Active Contour Segmentation')
plt.xlabel('X Label')
plt.ylabel('Y Label')

# Plotting the face boundary marker
ax.plot(astronaut_snake[:, 0],
        astronaut_snake[:, 1],
        '-b', lw=5)

# Plotting the circle around face
ax.plot(snake[:, 0], snake[:, 1], '--r', lw=5)
 
fig, axes = plt.subplots(1, 3, figsize=(10, 10))

# Sample Image of scikit-image package
astronaut = data.astronaut()
gray_astronaut = rgb2gray(astronaut)

# Computing the Chan VESE segmentation technique
chanvese_gray_astronaut = chan_vese(gray_astronaut,
                                    max_num_iter=100,
                                    extended_output=True)

ax = axes.flatten()

# Plotting the original image
ax[0].imshow(gray_astronaut, cmap="gray")
ax[0].set_title("Original Image")
ax[0].set_xlabel('X Label')
ax[0].set_ylabel('Y Label')

# Plotting the segmented - 100 iterations image
ax[1].imshow(chanvese_gray_astronaut[0], cmap="gray")
title = "Chan-Vese Segmentation - {} iterations".format(len(chanvese_gray_astronaut[2]))

ax[1].set_title(title)
ax[1].set_xlabel('X Label')
ax[1].set_ylabel('Y Label')

# Plotting the final level set
ax[2].imshow(chanvese_gray_astronaut[1], cmap="gray")
ax[2].set_title("Final Level Set")
ax[2].set_xlabel('X Label')
ax[2].set_ylabel('Y Label')
plt.show()

# Setting the plot figure as 15, 15
plt.figure(figsize=(15, 15))
 
# Sample Image of scikit-image package
astronaut = data.astronaut()

# Applying SLIC segmentation for the edges to be drawn over
astronaut_segments = slic(astronaut,
                        n_segments=100,
                        compactness=1)

plt.subplot(1, 2, 1)

# Plotting the original image
plt.imshow(astronaut)
plt.title('Original Image')
plt.xlabel('X Label')
plt.ylabel('Y Label')

# Detecting boundaries for labels
plt.subplot(1, 2, 2)

# Plotting the output of marked_boundaries function i.e. the image with segmented boundaries
plt.imshow(mark_boundaries(astronaut, astronaut_segments))
plt.title('SLIC Segmentation')
plt.xlabel('X Label')
plt.ylabel('Y Label')

# Setting the plot size as 15, 15
plt.figure(figsize=(15,15))

# Sample Image of scikit-image package
astronaut = data.astronaut()

# Applying Simple Linear Iterative
# Clustering on the image - 50 segments & compactness = 10
astronaut_segments = slic(astronaut,
                        n_segments=50,
                        compactness=10)
plt.subplot(1,2,1)

# Plotting the original image
plt.imshow(astronaut)
plt.title('Original Image')
plt.xlabel('X Label')
plt.ylabel('Y Label')

plt.subplot(1,2,2)

# Converts a label image into an RGB color image for visualizing the labeled regions
plt.imshow(label2rgb(astronaut_segments,
                    astronaut,
                    kind = 'avg'))
plt.title('SLIC Segmentation with Color Mapping')
plt.xlabel('X Label')
plt.ylabel('Y Label')

# Setting the figure size as 15, 15
plt.figure(figsize=(15,15))

# Sample Image of scikit-image package
astronaut = data.astronaut()

# computing the Felzenszwalb's Segmentation with sigma = 5 and minimum size = 100
astronaut_segments = felzenszwalb(astronaut,
                                scale = 2,
                                sigma=5,
                                min_size=100)

# Plotting the original image
plt.subplot(1,2,1)
plt.imshow(astronaut)
plt.title('Original Image')
plt.xlabel('X Label')
plt.ylabel('Y Label')

# Marking the boundaries of Felzenszwalb's segmentations
plt.subplot(1,2,2)
plt.imshow(mark_boundaries(astronaut,
                        astronaut_segments))
plt.title('Felzenszwalb Segmentations')
plt.xlabel('X Label')
plt.ylabel('Y Label')
