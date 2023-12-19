import cv2
import numpy as np
from sklearn.cluster import KMeans
from constraint import Problem
import random
import os
from matplotlib.colors import LinearSegmentedColormap

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def resize_image(image, scale_percent=20):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def k_means_segmentation(image, K=4):
    pixel_values = np.float32(image.reshape((-1, 1)))
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
    labels = kmeans.fit_predict(pixel_values)
    return labels.reshape(image.shape)

def create_coloring_csp(labels, color_count):
    height, width = labels.shape
    problem = Problem()

    segments = np.unique(labels)
    for segment in segments:
        problem.addVariable(segment, range(color_count))

    for y in range(height):
        for x in range(width):
            if x + 1 < width and labels[y, x] != labels[y, x + 1]:
                problem.addConstraint(lambda a, b: a != b, [labels[y, x], labels[y, x + 1]])
            if y + 1 < height and labels[y, x] != labels[y + 1, x]:
                problem.addConstraint(lambda a, b: a != b, [labels[y, x], labels[y + 1, x]])

    return problem.getSolution()

def generate_random_color():
    # Generate a random color using HSV model for more variety
    hue = np.random.rand() * 360  # Hue value between 0-360
    saturation = 0.5 + np.random.rand() * 0.5  # Saturation between 0.5-1.0 for more vivid colors
    value = 0.5 + np.random.rand() * 0.5  # Value between 0.5-1.0 to avoid very dark colors
    return cv2.cvtColor(np.uint8([[[hue, saturation * 255, value * 255]]]), cv2.COLOR_HSV2BGR)[0][0]

def generate_gradient_colors(color_count):
    colors = []
    for _ in range(color_count):
        start_color = generate_random_color()
        end_color = generate_random_color()
        colors.append((start_color, end_color))
    return colors

def interpolate_color(start_color, end_color, fraction):
    return start_color + (end_color - start_color) * fraction

def apply_gradient_color_to_solution(image, labels, solution, colors):
    if not solution:
        return None

    result_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            color_index = solution[labels[y, x]]
            start_color, end_color = colors[color_index]
            fraction = np.mean([y / image.shape[0], x / image.shape[1]])
            color = interpolate_color(start_color, end_color, fraction)
            result_image[y, x] = np.clip(color, 0, 255).astype(np.uint8)

    return result_image

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

def mark_face_regions(labels, faces, special_label):
    for (x, y, w, h) in faces:
        labels[y:y+h, x:x+w] = special_label
    return labels

def process_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File at {image_path} not found or cannot be read.")

    img = cv2.imread(image_path)
    faces = detect_faces(img)

    grayscale_img = convert_to_grayscale(img)
    blurred_img = apply_gaussian_blur(grayscale_img)
    resized_img = resize_image(blurred_img)
    labels = k_means_segmentation(resized_img, K=8)

    # Tandai area wajah dengan label khusus
    special_label = 999  # Label unik untuk area wajah
    labels = mark_face_regions(labels, faces, special_label)

    color_count = 15
    colors = generate_gradient_colors(color_count)
    solution = create_coloring_csp(labels, color_count)
    colored_img = apply_gradient_color_to_solution(resized_img, labels, solution, colors)

    if colored_img is not None:
        result_img = cv2.resize(colored_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite('static/files/coloring_csp_image.png', result_img)
    else:
        print("No solution found")