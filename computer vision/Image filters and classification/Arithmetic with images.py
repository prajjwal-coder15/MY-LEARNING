"""
Arithmetic with images
"""

# Import required packages:
import numpy as np
import cv2
import matplotlib.pyplot as plt


def show_with_matplotlib(color_img, title, pos):
    """Shows an images using matplotlib capabilities"""

    # Convert BGR images to RGB:
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Create the dimensions of the figure and set title:
plt.figure(figsize=(12, 6))
plt.suptitle("Arithmetic with images", fontsize=14, fontweight='bold')

# Load the original images:
image = cv2.imread('test1.png')

# Add 60 to every pixel on the images. The result will look lighter:
M = np.ones(image.shape, dtype="uint8") * 60
added_image = cv2.add(image, M)

# Subtract 60 from every pixel. The result will look darker:
subtracted_image = cv2.subtract(image, M)

# Additionally, we can build an scalar and add/subtract it:
scalar = np.ones((1, 3), dtype="float") * 110
added_image_2 = cv2.add(image, scalar)
subtracted_image_2 = cv2.subtract(image, scalar)

# Display all the resulting images:
show_with_matplotlib(image, "images", 1)
show_with_matplotlib(added_image, "added 60 (images + images)", 2)
show_with_matplotlib(subtracted_image, "subtracted 60 (images - images)", 3)
show_with_matplotlib(added_image_2, "added 110 (images + scalar)", 5)
show_with_matplotlib(subtracted_image_2, "subtracted 110 (images - scalar)", 6)

# Show the Figure:
plt.show()