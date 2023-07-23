import cv2
import matplotlib.pyplot as plt

# Read the original image in grayscale
img_gray = cv2.imread('Lighthouse.jpg', cv2.IMREAD_GRAYSCALE)

# apply thresholding on gray image
ret, thresh = cv2.threshold(img_gray, 150, 255, 0)

# Find the contours in the image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Print the number of detected contours
print("Number of Contours detected:", len(contours))

# Draw the first contour
plt.imshow(img_gray, cmap='gray')
plt.title('Contour: 1') # Set the title of the plot
plt.plot(contours[0][:, 0, 0], contours[0][:, 0, 1], '-g', linewidth=2) # Plot the first contour
plt.show() # Display the plot

# print the moments of the first contour
cnt = contours[0] # Get the first contour
M = cv2.moments(cnt) # Compute the moments of the first contour
print("Moments of first contour:", M) # Print the moments
