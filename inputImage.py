import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load Image A
image_path = "airbnb.webp"  # Replace with the path to your image
image_a = cv2.imread(image_path)

# Display the image
plt.imshow(image_a)
plt.axis('off')  # Remove axes for better display
plt.show()