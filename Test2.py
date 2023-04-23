
import cv2
import numpy as np

def find_squares(image_path):
    # Load image and convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    cv2.imshow("Blur", blurred)

    # Define a list of adaptive threshold parameters to try
    threshold_params = [(11, 2), (13, 4), (15, 6)]

    for block_size, c in threshold_params:
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c)

        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        thresh = cv2.erode(thresh, kernel, iterations=2)
        #thresh = cv2.dilate(thresh, kernel, iterations=1)

        cv2.imshow("Thresh", thresh)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours to find squares
        squares = []
        for cnt in contours:
            # Approximate contour shape
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Check if the contour has 4 vertices, is convex, and has a relatively large area
            if len(approx) == 4 and cv2.isContourConvex(approx) and cv2.contourArea(approx) > 250 and cv2.contourArea(approx) < 10000:
                squares.append(approx)

        # If enough squares are found, stop searching
        if len(squares) >= 16:
            break

    return squares

def group_squares_into_rows(squares, tolerance=10):
    squares_by_y = sorted(squares, key=lambda x: cv2.boundingRect(x)[1])
    rows = []
    current_row = []
    last_y = None

    for square in squares_by_y:
        y = cv2.boundingRect(square)[1]
        if last_y is None or abs(y - last_y) <= tolerance:
            current_row.append(square)
        else:
            current_row.sort(key=lambda x: cv2.boundingRect(x)[0])
            rows.append(current_row)
            current_row = [square]
        last_y = y

    if current_row:
        current_row.sort(key=lambda x: cv2.boundingRect(x)[0])
        rows.append(current_row)

    return rows

def single_frame():
    image_path = "c:/Users/12247/Desktop/GeomAI-Test/image5.png"
    squares = find_squares(image_path)

    # Group squares into rows and sort them from left to right
    square_matrix = group_squares_into_rows(squares)

    # Display the results
    image = cv2.imread(image_path)
    for row in square_matrix:
        for square in row:
            cv2.drawContours(image, [square], -1, (0, 255, 0), 2)
            cv2.imshow("Squares", image)
    cv2.imshow("Squares", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main_webcam():
    # Open the webcam
    cap = cv2.VideoCapture("c:/Users/12247/Desktop/GeomAI-Test/A32Real1.mp4")

    if not cap.isOpened():
        print("Error: Unable to open the webcam.")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    size = (frame_width, frame_height)
    
    result = cv2.VideoWriter('c:/Users/12247/Desktop/GeomAI-Test/output.avi', cv2.VideoWriter_fourcc(*'MJPG'),30, size)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame was successfully captured
        if not ret:
            print("Error: Unable to read the frame.")
            break

        # Save the captured frame as an image
        cv2.imwrite("temp_frame.jpg", frame)

        # Process the image to find squares
        squares = find_squares("temp_frame.jpg")

        # Group squares into rows and sort them from left to right
        square_matrix = group_squares_into_rows(squares)

        # Draw the squares on the frame and display it
        for row in square_matrix:
            for square in row:
                cv2.drawContours(frame, [square], -1, (0, 255, 0), 2)
                cv2.imshow("Webcam Output", frame)
                
        #cv2.waitKey(0)

        cv2.imshow("Webcam Output", frame)

        result.write(frame)

        # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture and close all windows
    cap.release()
    result.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_webcam()
    #single_frame()