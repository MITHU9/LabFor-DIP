import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the input image (grayscale for simplicity)
image_path = "airbnb.webp"  # Replace with your image path
image_a = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image_a is None:
    raise ValueError("Image not found! Please check the image path.")

# Display the original image
plt.figure(figsize=(6, 6))
plt.imshow(image_a, cmap='gray')
plt.title("Original Image A")
plt.axis('off')
plt.show()

# Step 2: Compute and draw histogram manually
def compute_histogram(image):
    histogram = np.zeros(256)
    for pixel in image.ravel():
        histogram[pixel] += 1
    return histogram

hist_a = compute_histogram(image_a)

# Plot histogram
plt.figure(figsize=(8, 6))
plt.bar(range(256), hist_a, color='gray')
plt.title("Histogram of Original Image A")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()

# Step 3: Histogram equalization
def histogram_equalization(image):
    hist = compute_histogram(image)
    cdf = hist.cumsum()  # Cumulative distribution function
    cdf_normalized = cdf / cdf[-1]  # Normalize to [0, 1]
    equalized_image = np.floor(cdf_normalized[image] * 255).astype(np.uint8)
    return equalized_image

equalized_image_a = histogram_equalization(image_a)

# Display equalized image
plt.figure(figsize=(6, 6))
plt.imshow(equalized_image_a, cmap='red')
plt.title("Equalized Image A")
plt.axis('off')
plt.show()

# Step 4: Compute and draw histogram of equalized image
hist_equalized = compute_histogram(equalized_image_a)

plt.figure(figsize=(8, 6))
plt.bar(range(256), hist_equalized, color='gray')
plt.title("Histogram of Equalized Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()

# Step 5: Apply high-pass filter (e.g., Laplacian kernel)
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])
filtered_image_a = cv2.filter2D(image_a, -1, kernel)

# Display filtered image
plt.figure(figsize=(6, 6))
plt.imshow(filtered_image_a, cmap='gray')
plt.title("High-Pass Filtered Image A")
plt.axis('off')
plt.show()

# Step 6: Compute and draw histogram of filtered image
hist_filtered = compute_histogram(filtered_image_a)

plt.figure(figsize=(8, 6))
plt.bar(range(256), hist_filtered, color='gray')
plt.title("Histogram of Filtered Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()
