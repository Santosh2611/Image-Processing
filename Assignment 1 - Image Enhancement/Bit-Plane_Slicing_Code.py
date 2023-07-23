import cv2  # Wrapper package for OpenCV python bindings
import numpy as np  # General-purpose array-processing package

# Read the input image
img = cv2.imread('Lighthouse.jpg', 0)

# Convert each pixel value to binary and store it in a list
lst = [np.binary_repr(pixel, width=8) for pixel in img.flatten()]

# Initialize empty lists for each bit plane
bit_planes = [[] for _ in range(8)]

# Store the bit value of each pixel in its corresponding bit plane
for i in range(8):
    for j in range(len(lst)):
        bit_planes[i].append(int(lst[j][i]))

    # Multiply with 2^(n-1) and reshape to reconstruct the bit image
    bit_planes[i] = (np.array(bit_planes[i], dtype=np.uint8) * 2**(7-i)).reshape(img.shape)

# Concatenate the bit planes horizontally and vertically for ease of display
finalr = cv2.hconcat(bit_planes[:4])
finalv = cv2.hconcat(bit_planes[4:])
final = cv2.vconcat([finalr, finalv])

# Write the bit planes and the combined 4-bit plane images
cv2.imwrite('8 Bit Planes - Lighthouse.jpg', final)
cv2.imwrite('Image Using 4 Bit Planes - Lighthouse.jpg', bit_planes[0] + bit_planes[1] + bit_planes[2] + bit_planes[3])
