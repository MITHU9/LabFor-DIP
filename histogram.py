import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load Image A
image_path = "airbnb.webp"  # Replace with the path to your image
image_a = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Convert to grayscale

# Convert image to numpy array
image_array = np.array(image_a)

# Calculate histogram
histogram = [0] * 256  # 256 bins for grayscale values (0-255)

for row in image_array:
    for pixel in row:
        histogram[pixel] += 1

# Normalize the histogram (optional)
max_count = max(histogram)
normalized_histogram = [count / max_count for count in histogram]

# Plot the histogram
plt.bar(range(256), normalized_histogram, color='blue')
plt.title("Histogram of Image A")
plt.xlabel("Pixel Value")
plt.ylabel("Normalized Frequency")
plt.show()
