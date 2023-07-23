import warnings
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2gray

warnings.filterwarnings('ignore')

def get_entropy_image(image_gray, radius):
    return entropy(image_gray, disk(radius))

def plot_disk_iterations(image_gray):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    radii = range(1, 10)
    fig.suptitle('Disk Iterations', fontsize=24)
    
    for radius, ax in zip(radii, axes.flatten()):
        ax.set_title(f'Radius at {radius}', fontsize=20)
        
        entropy_image = get_entropy_image(image_gray, radius)
        ax.imshow(entropy_image, cmap='magma')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')

    fig.tight_layout()
    return fig, axes

    
def plot_threshold_checker(image_gray):
    entropy_image = get_entropy_image(image_gray, 6)
    scaled_entropy = entropy_image / entropy_image.max()  
    fig, axes = plt.subplots(2, 5, figsize=(17, 10))
    thresholds = [0.1 * (i+1) for i in range(10)]
    fig.suptitle('Threshold Checker', fontsize=24)
    
    for threshold, ax in zip(thresholds, axes.flatten()):
        ax.set_title(f'Threshold: {threshold:.2f}', fontsize=16)
        threshold_mask = scaled_entropy > threshold
        ax.imshow(threshold_mask, cmap='gist_stern_r') 
        ax.axis('off')

    fig.tight_layout()
    return fig, axes


def plot_entropy_mask_viz(image_gray):
    entropy_image = get_entropy_image(image_gray, 6)
    scaled_entropy = entropy_image / entropy_image.max()

    fig, axes = plt.subplots(1, 2, figsize=(17, 10))
    f_size = 24
    titles = ['Greater Than Threshold', 'Less Than Threshold']
    
    for condition, title, ax in zip([scaled_entropy > 0.8, scaled_entropy < 0.8], titles, axes.flatten()):
        masked_image = np.stack([image_gray]*3, axis=-1)
        masked_image[~condition] = 0
        
        ax.imshow(masked_image)
        ax.axis('off')
        ax.set_title(title, fontsize=f_size)

    fig.suptitle('Entropy Mask Visualization', fontsize=24)
    fig.tight_layout()
    return fig, axes

# Read image and save to variable for later use
shawls = io.imread('Lighthouse.jpg')

# Convert to grayscale and unsigned byte
shawl_gray = img_as_ubyte(rgb2gray(shawls))

# Show original and grayscale images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
titles = ['Original Image', 'Grayscale Image']
images = [shawls, shawl_gray]

for ax, image, title in zip(axes.flatten(), images, titles):
    ax.imshow(image)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_title(title, fontsize=20)

plt.tight_layout()

# Calculate entropy image and show it
entropy_image = get_entropy_image(shawl_gray, 6)
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.imshow(entropy_image, cmap='magma')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_title('Entropy Image', fontsize=20)
plt.tight_layout()

# Show entropic images for different disk radii
plot_disk_iterations(shawl_gray)

# Show thresholded versions of the entropic image
plot_threshold_checker(shawl_gray)

# Show masked grayscale image based on entropy threshold
plot_entropy_mask_viz(shawl_gray)

plt.show()
