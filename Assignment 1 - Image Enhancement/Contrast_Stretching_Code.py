from PIL import Image # Provides a class with the same name which is used to represent a PIL image
from matplotlib import pyplot as plt # Collection of command style functions that make matplotlib work like MATLAB

# Define methods to process the red, green, and blue bands of the image
def normalize(intensity, minI, maxI, minO, maxO):
    # Scale pixel values from input range (minI, maxI) to output range (minO, maxO)
    return (intensity - minI) * ((maxO - minO) / (maxI - minI)) + minO

# Read the input image and split it into red, green, and blue bands
image = Image.open("Lighthouse.jpg")
bands = image.split()

# Define the minimum and maximum intensity values for each color band
red_min, red_max = 86, 230
green_min, green_max = 90, 225
blue_min, blue_max = 100, 210

# Apply contrast stretching on each color band
normalized_red = bands[0].point(lambda i: normalize(i, red_min, red_max, 0, 255))
normalized_green = bands[1].point(lambda i: normalize(i, green_min, green_max, 0, 255))
normalized_blue = bands[2].point(lambda i: normalize(i, blue_min, blue_max, 0, 255))

# Merge the resulting red, green, and blue bands into a new image
normalized_image = Image.merge("RGB", (normalized_red, normalized_green, normalized_blue))

# Display the resulting image
plt.imshow(normalized_image)
plt.axis('off')
plt.show()
