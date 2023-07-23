import cv2
import matplotlib.pyplot as plt

# Read the original image in grayscale
img_gray = cv2.imread('Lighthouse.jpg', cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    print("Error: Could not read image file")
else:
    # Display original image
    plt.imshow(cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB colorspace for display in matplotlib
    plt.title('Original')
    plt.axis('off')
    plt.show()

    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection

    # Display Sobel Edge Detection Images using subplots
    fig, axes = plt.subplots(nrows=1, ncols=3)  # Create a 1x3 grid of plots
    titles = ['Sobel X', 'Sobel Y', 'Sobel X Y using Sobel() function']
    images = [sobelx, sobely, sobelxy]
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')  # Display each sobel image on a different subplot
        ax.set_title(titles[i])
        ax.axis('off')
    plt.tight_layout()  # Adjust the spacing between subplots for better visibility
    plt.show()

    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection using img_blur
    # Display Canny Edge Detection Image
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')
    plt.show()
