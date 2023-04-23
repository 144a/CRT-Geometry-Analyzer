import cv2
import numpy as np
from scipy.interpolate import splprep, splev


def pre_process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow("thresh", threshold)
    return threshold


def detect_edges(image):
    return cv2.Canny(image, 50, 200)


def find_contours(edge_image):
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def fit_cubic_spline(contour, smoothing=0.1):
    points = contour[:, 0, :]
    tck, u = splprep([points[:, 0], points[:, 1]], s=smoothing, k=3)
    spline_points = np.column_stack(splev(u, tck)).astype(np.int32)
    return spline_points


def is_vertical_line(points):
    rect = cv2.minAreaRect(points)
    width, height = rect[1]

    return height >= width


def process_image(image_path, min_contour_length=30):
    pre_processed_image = pre_process_image(image_path)
    edge_image = detect_edges(pre_processed_image)
    contours = find_contours(edge_image)

    image_color = cv2.cvtColor(pre_processed_image, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        contour_length = cv2.arcLength(contour, closed=False)

        if len(contour) > 3 and contour_length >= min_contour_length:
            cubic_spline_points = fit_cubic_spline(contour)

            if is_vertical_line(cubic_spline_points):
                line_color = (0, 0, 255)  # Red for vertical lines
            else:
                line_color = (0, 255, 0)  # Green for horizontal lines

            cv2.polylines(image_color, [cubic_spline_points], False, line_color, 2)

    return image_color


if __name__ == "__main__":
    input_image_path = "image4.png"
    output_image_path = "output_image.png"

    result_image = process_image(input_image_path)
    cv2.imwrite(output_image_path, result_image)