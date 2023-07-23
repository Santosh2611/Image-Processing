import cv2
import matplotlib.pyplot as plt

# Read the original image and know its type
original_image = cv2.imread('Lighthouse.jpg', 0)

# Obtain the size of the original image
height, width = original_image.shape

# Show the original image
plt.imshow(original_image, cmap="gray")
plt.title("Original Image")
plt.xlabel("Width")
plt.ylabel("Height")
plt.show()

# Down sampling
downsampling_rate = 4
downsampled_image = cv2.resize(original_image, (width // downsampling_rate, height // downsampling_rate))

# Show down-sampled image
plt.imshow(downsampled_image, cmap="gray")
plt.title("Down Sampled Image")
plt.xlabel("Width")
plt.ylabel("Height")
plt.show()

# Up-sampling
upsampled_image = cv2.resize(downsampled_image, (width, height), interpolation=cv2.INTER_NEAREST)

# Plot the up-sampled image
plt.imshow(upsampled_image, cmap="gray")
plt.title("Up Sampled Image")
plt.xlabel("Width")
plt.ylabel("Height")
plt.show()
