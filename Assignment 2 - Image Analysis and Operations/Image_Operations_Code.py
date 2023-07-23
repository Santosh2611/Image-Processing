import numpy as np
from concurrent import futures
from matplotlib import pyplot as plt
from skimage import io
import cv2

# Read the image
def read_image(filename):
    return io.imread(filename)


image = read_image("Lighthouse.jpg")


# Create a kernel
def create_kernel(size):
    return np.ones((size, size), np.uint8)


kernel = create_kernel(5)


# Erode the image with kernel
def erode_image(image, kernel):
    return cv2.erode(image, kernel)


# Dilate the image with kernel
def dilate_image(image, kernel, iterations=1):
    return cv2.dilate(image, kernel, iterations=iterations)


# Open the image with kernel
def open_image(image, kernel):
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# Denoise the image using fastNlMeansDenoisingColored() method
def denoise_image(image, denoising_parameters):
    return cv2.fastNlMeansDenoisingColored(image, None, *denoising_parameters)


# Add a border to the image using copyMakeBorder() method
def add_border(image, top, bottom, left, right, border_type, value):
    return cv2.copyMakeBorder(image, top, bottom, left, right, border_type, None, value=value)


# Convert the image to grayscale and calculate its histogram
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Plot the histogram of the color image
def plot_histogram(image):
    histr = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(histr)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')


# Plot the individual color histograms
def plot_color_histograms(image):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')


# Define parameters for denoising_image function
denoising_parameters = (15, 8, 8, 15)


# Execute functions concurrently
with futures.ThreadPoolExecutor() as executor:
    image_erode = executor.submit(erode_image, image, kernel)
    image_dilation = executor.submit(dilate_image, image, kernel)
    image_erode_dilate = executor.submit(open_image, image, kernel)
    denoised_image = executor.submit(denoise_image, image, denoising_parameters)

# Plot original image and processed images
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(image)
axs[0, 0].set_xlabel('X-axis')
axs[0, 0].set_ylabel('Y-axis')
axs[0, 0].set_title('Original Image')

axs[0, 1].imshow(image_erode.result())
axs[0, 1].set_xlabel('X-axis')
axs[0, 1].set_ylabel('Y-axis')
axs[0, 1].set_title('Eroded Image')

axs[1, 0].imshow(image_dilation.result())
axs[1, 0].set_xlabel('X-axis')
axs[1, 0].set_ylabel('Y-axis')
axs[1, 0].set_title('Dilated Image')

axs[1, 1].imshow(image_erode_dilate.result())
axs[1, 1].set_xlabel('X-axis')
axs[1, 1].set_ylabel('Y-axis')
axs[1, 1].set_title('Eroded and Dilated Image')

plt.show()


# Plot images with border
image_border1 = add_border(image, 25, 25, 10, 10, cv2.BORDER_CONSTANT, 0)
plt.imshow(image_border1)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Image with Black Border')
plt.show()

image_border2 = add_border(image, 250, 250, 250, 250, cv2.BORDER_REFLECT, value=None)
plt.imshow(image_border2)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Image with Mirrored Border')
plt.show()

image_border3 = add_border(image, 300, 250, 100, 50, cv2.BORDER_REFLECT, value=None)
plt.imshow(image_border3)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Image with Modified Mirrored Border')
plt.show()


# Plot denoised image
plt.imshow(denoised_image.result())
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Denoised Image')
plt.show()


# Plot histogram of color image
plt.subplot(1,2,1)
plot_histogram(image)
plt.subplot(1,2,2)
plt.hist(image.ravel(), 256, [0, 256])
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.suptitle('Histogram of the Color Image')
plt.tight_layout()
plt.show()

# Plot grayscale image and its histogram
grey_image = convert_to_grayscale(image)
histogram = cv2.calcHist([grey_image], [0], None, [256], [0, 256])

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(grey_image, cmap='gray')
axs[0].set_xlabel('X-axis')
axs[0].set_ylabel('Y-axis')
axs[0].set_title('Grayscale Image')

axs[1].plot(histogram, color='k')
axs[1].set_xlabel('Pixel Value')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Histogram of Grayscale Image')

plt.tight_layout()
plt.show()


# Plot individual color histograms
plot_color_histograms(image)
plt.title('Individual Color Histograms')
plt.show()
