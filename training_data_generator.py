import cv2
import numpy as np
from scipy.interpolate import splprep, splev


def pre_process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #blurred = cv2.GaussianBlur(image, (15, 15), 0)
    #cv2.imshow("blurred", blurred)
    _, threshold = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("threshold", threshold)
    return threshold


def detect_edges(image):
    return cv2.Canny(image, 50, 200)


def find_contours(edge_image):
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.fillPoly(edge_image, pts=contours, color=(255,255,255))
    #contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def fit_cubic_spline(contour, smoothing=0.1):
    points = contour[:, 0, :]
    tck, u = splprep([points[:, 0], points[:, 1]], s=smoothing, k=3)
    spline_points = np.column_stack(splev(u, tck)).astype(np.int32)
    return spline_points


def process_image(image_path, min_contour_length=10):
    pre_processed_image = pre_process_image(image_path)
    edge_image = detect_edges(pre_processed_image)
    contours = find_contours(edge_image)

    total = 0

    for contour in contours:
        contour_length = cv2.arcLength(contour, closed=False)

        if len(contour) > 3 and contour_length >= min_contour_length:
            total += 1
            cubic_spline_points = fit_cubic_spline(contour)
            cv2.polylines(edge_image, [cubic_spline_points], False, 255, 2)
    print("A total of ", total)
    return edge_image


if __name__ == "__main__":
    input_image_path = "image.png"
    output_image_path = "output_image.png"

    result_image = process_image(input_image_path)
    cv2.imwrite(output_image_path, result_image)
    cv2.waitKey(0)