import cv2
import numpy as np
from sklearn.cluster import KMeans
from constraint import Problem
import random

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

    # Custom coloring constraints
    for y in range(height):
        for x in range(width):
            if x + 1 < width and labels[y, x] != labels[y, x + 1]:
                problem.addConstraint(lambda a, b: a != b, [labels[y, x], labels[y, x + 1]])
            if y + 1 < height and labels[y, x] != labels[y + 1, x]:
                problem.addConstraint(lambda a, b: a != b, [labels[y, x], labels[y + 1, x]])

    return problem.getSolution()

def generate_colors(color_count):
    colors = []
    while len(colors) < color_count:
        new_color = tuple(random.randint(0, 255) for _ in range(3))
        colors.append(new_color)
    return colors

def apply_color_to_solution(image, labels, solution, colors):
    if not solution:
        return None

    result_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            color_index = solution[labels[y, x]]
            result_image[y, x] = colors[color_index]

    return result_image

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"File at {image_path} not found or cannot be read.")

    grayscale_img = convert_to_grayscale(img)
    blurred_img = apply_gaussian_blur(grayscale_img)
    resized_img = resize_image(blurred_img)
    labels = k_means_segmentation(resized_img, K=8)  # Increased number of segments
    color_count = 15  # Increased number of colors for more complexity
    colors = generate_colors(color_count)
    solution = create_coloring_csp(labels, color_count)
    colored_img = apply_color_to_solution(resized_img, labels, solution, colors)

    if colored_img is not None:
        result_img = cv2.resize(colored_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite('static/files/coloring_csp_image.png', result_img)
    else:
        print("No solution found")