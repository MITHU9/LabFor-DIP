import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load Image A
image_path = "airbnb.webp"  # Replace with the path to your image
image_a = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Convert to grayscale

hist = np.zeros(256 // 10 + 1, dtype=np.int32)

rows, cols = image_a.shape[:2]


# Compute histogram
for i in range(rows):
    for j in range(cols):
        pixel = image_a[i][j] // 10
        hist[pixel] += 1


x = np.arange(256 // 10 + 1) * 10


plt.figure(figsize=(16,10))
plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(image_a,cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.figure(figsize=(8, 6))
plt.subplot(2,2,2)
plt.bar(x, hist, width=5, align="edge", color='gray',alpha=1)
plt.title("Histogram of Image A")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()

filtered = cv2.GaussianBlur(image_a,(5,5),0)
hist_fil = np.zeros(256 // 10 + 1, dtype=np.int32)
for i in range(rows):
    for j in range(cols):
        pixel= filtered[i][j]//10
        hist_fil[pixel]+=1


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

# Compute histogram of filtered image
plt.subplot(2,3,5)
plt.bar(x, hist_fil, width=5, align='edge', color='black', alpha=1)
plt.title(" Filtered Histogram with Bins of Size 5")
plt.xlabel("Pixel Intensity Range")
plt.ylabel("Frequency")
plt.show()







# # 3. Binary Image
# _, binary_img = cv2.threshold(image_a, 128, 255, cv2.THRESH_BINARY)

# plt.figure(figsize=(8, 6))
# plt.imshow(binary_img, cmap='gray')
# plt.title("Binary Image")
# plt.axis('off')
# plt.show()

# # 4. Image Negative
# negative_img = 255 - image_a

# plt.figure(figsize=(8, 6))
# plt.imshow(negative_img, cmap='gray')
# plt.title("Negative Image")
# plt.axis('off')
# plt.show()

# # 5. Log Transform
# c = 255 / (np.log(1 + np.max(image_a)))
# log_img = (c * np.log(1 + image_a)).astype(np.uint8)

# plt.figure(figsize=(8, 6))
# plt.imshow(log_img, cmap='gray')
# plt.title("Log Transform")
# plt.axis('off')
# plt.show()

# # 6. Power Transform
# gamma = 2.0  # Example gamma value
# normalized = image_a / 255.0
# power_img = (np.power(normalized, gamma) * 255).astype(np.uint8)

# plt.figure(figsize=(8, 6))
# plt.imshow(power_img, cmap='gray')
# plt.title(f"Power Transform (Gamma = {gamma})")
# plt.axis('off')
# plt.show()

# # 7. Blurring (Gaussian Blur)
# blurred_img = cv2.GaussianBlur(image_a, (5, 5), 0)

# plt.figure(figsize=(8, 6))
# plt.imshow(blurred_img, cmap='gray')
# plt.title("Blurred Image (Gaussian Blur)")
# plt.axis('off')
# plt.show()

# # 8. Edge Detection
# edges = cv2.Canny(image_a, 100, 200)

# plt.figure(figsize=(8, 6))
# plt.imshow(edges, cmap='gray')
# plt.title("Edge Detection")
# plt.axis('off')
# plt.show()

# # 9. Erosion
# kernel = np.ones((3, 3), np.uint8)
# eroded_img = cv2.erode(binary_img, kernel)

# plt.figure(figsize=(8, 6))
# plt.imshow(eroded_img, cmap='gray')
# plt.title("Erosion")
# plt.axis('off')
# plt.show()

# # 10. Dilation
# dilated_img = cv2.dilate(binary_img, kernel)

# plt.figure(figsize=(8, 6))
# plt.imshow(dilated_img, cmap='gray')
# plt.title("Dilation")
# plt.axis('off')
# plt.show()

# # 11. Opening (Erosion followed by Dilation)
# opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

# plt.figure(figsize=(8, 6))
# plt.imshow(opened_img, cmap='gray')
# plt.title("Opening (Morphological)")
# plt.axis('off')
# plt.show()

# # 12. Closing (Dilation followed by Erosion)
# closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

# plt.figure(figsize=(8, 6))
# plt.imshow(closed_img, cmap='gray')
# plt.title("Closing (Morphological)")
# plt.axis('off')
# plt.show()

# # 13. Histogram Equalization
# equalized_img = cv2.equalizeHist(image_a)

# plt.figure(figsize=(8, 6))
# plt.imshow(equalized_img, cmap='gray')
# plt.title("Histogram Equalized Image")
# plt.axis('off')
# plt.show()


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def plot_image(title, img, cmap='gray'):
#     """Helper function to display an image."""
#     plt.figure(figsize=(6, 6))
#     plt.title(title)
#     plt.axis('off')
#     plt.imshow(img, cmap=cmap)
#     plt.show()

# # 1. Image Input/Output
# def load_and_plot_image(image_path):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     plot_image("Loaded Image", img)
#     return img

# # 2. Binary Image
# def binary_image(img, threshold=128):
#     _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
#     plot_image("Binary Image", binary)
#     return binary

# # 3. Boundary Sum
# def boundary_sum(binary_img):
#     rows, cols = binary_img.shape
#     return binary_img[0, :].sum() + binary_img[-1, :].sum() + binary_img[:, 0].sum() + binary_img[:, -1].sum()

# # 4. Diagonal Sum
# def diagonal_sum(binary_img):
#     main_diag = np.trace(binary_img)
#     anti_diag = np.trace(np.fliplr(binary_img))
#     return main_diag + anti_diag

# # 5. Histogram
# def image_histogram(img):
#     plt.figure(figsize=(6, 4))
#     plt.hist(img.ravel(), 256, [0, 256])
#     plt.title("Image Histogram")
#     plt.xlabel("Pixel Intensity")
#     plt.ylabel("Frequency")
#     plt.show()

# # 6. Image Negative
# def image_negative(img):
#     negative = 255 - img
#     plot_image("Negative Image", negative)
#     return negative

# # 7. Log Transform
# def log_transform(img):
#     c = 255 / (np.log(1 + np.max(img)))
#     log_img = (c * np.log(1 + img)).astype(np.uint8)
#     plot_image("Log Transformed Image", log_img)
#     return log_img

# # 8. Power Transform
# def power_transform(img, gamma):
#     normalized = img / 255.0
#     power_img = (np.power(normalized, gamma) * 255).astype(np.uint8)
#     plot_image(f"Power Transform (Gamma={gamma})", power_img)
#     return power_img

# # 9. Split and Merge
# def split_and_merge(img):
#     b, g, r = cv2.split(img)
#     merged_img = cv2.merge((b, g, r))
#     plot_image("Merged Image", cv2.cvtColor(merged_img, cv2.COLOR_BGR2RGB), cmap=None)
#     return merged_img

# # 10. Padding
# def add_padding(img, padding_size, padding_value=0):
#     padded_img = cv2.copyMakeBorder(img, padding_size, padding_size, padding_size, padding_size, 
#                                     cv2.BORDER_CONSTANT, value=padding_value)
#     plot_image("Padded Image", padded_img)
#     return padded_img

# # 11. Image Blurring
# def image_blurring(img, kernel_size=(5, 5)):
#     blurred_img = cv2.GaussianBlur(img, kernel_size, 0)
#     plot_image("Blurred Image", blurred_img)
#     return blurred_img

# # 12. Image Sharpening
# def image_sharpening(img):
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#     sharpened_img = cv2.filter2D(img, -1, kernel)
#     plot_image("Sharpened Image", sharpened_img)
#     return sharpened_img

# # 13. Edge Detection
# def edge_detection(img):
#     edges = cv2.Canny(img, 100, 200)
#     plot_image("Edge Detected Image", edges)
#     return edges

# # 14. Line Detection
# def line_detection(img):
#     edges = cv2.Canny(img, 50, 150)
#     lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
#     line_img = np.copy(img)
#     if lines is not None:
#         for rho, theta in lines[:, 0]:
#             a, b = np.cos(theta), np.sin(theta)
#             x0, y0 = a * rho, b * rho
#             x1 = int(x0 + 1000 * (-b))
#             y1 = int(y0 + 1000 * (a))
#             x2 = int(x0 - 1000 * (-b))
#             y2 = int(y0 - 1000 * (a))
#             cv2.line(line_img, (x1, y1), (x2, y2), 255, 2)
#     plot_image("Line Detected Image", line_img)
#     return line_img

# # 15. Point Detection
# def point_detection(img):
#     kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
#     points = cv2.filter2D(img, -1, kernel)
#     plot_image("Point Detected Image", points)
#     return points

# # 16. Histogram Equalization
# def histogram_equalization(img):
#     equalized = cv2.equalizeHist(img)
#     plot_image("Histogram Equalized Image", equalized)
#     return equalized

# # 17. Boundary Extraction
# def boundary_extraction(binary_img):
#     kernel = np.ones((3, 3), np.uint8)
#     eroded = cv2.erode(binary_img, kernel)
#     boundary = binary_img - eroded
#     plot_image("Boundary Extracted Image", boundary)
#     return boundary

# # 18. Erosion
# def erosion(binary_img):
#     kernel = np.ones((3, 3), np.uint8)
#     eroded = cv2.erode(binary_img, kernel)
#     plot_image("Eroded Image", eroded)
#     return eroded

# # 19. Dilation
# def dilation(binary_img):
#     kernel = np.ones((3, 3), np.uint8)
#     dilated = cv2.dilate(binary_img, kernel)
#     plot_image("Dilated Image", dilated)
#     return dilated

# # 20. Opening
# def opening(binary_img):
#     kernel = np.ones((3, 3), np.uint8)
#     opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
#     plot_image("Opened Image", opened)
#     return opened

# # 21. Closing
# def closing(binary_img):
#     kernel = np.ones((3, 3), np.uint8)
#     closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
#     plot_image("Closed Image", closed)
#     return closed

# # 22. Pattern Recognition
# def pattern_recognition(img, template):
#     result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
#     _, _, _, max_loc = cv2.minMaxLoc(result)
#     top_left = max_loc
#     h, w = template.shape
#     matched_img = cv2.rectangle(img.copy(), top_left, (top_left[0] + w, top_left[1] + h), 255, 2)
#     plot_image("Pattern Recognized Image", matched_img)
#     return matched_img
