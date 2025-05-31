import cv2
import numpy as np

# Open video capture (0 = default webcam; you can also use a video file path)
cap = cv2.VideoCapture(0)

while True:
    # Read frame from video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV range for yellow color
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create mask where yellow colors are white and rest are black
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours of the masked yellow regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Ignore small areas to reduce noise
        area = cv2.contourArea(cnt)
        if area > 500:
            # Get bounding rectangle coordinates
            x, y, w, h = cv2.boundingRect(cnt)

            # Ensure the box is square by using max(w, h)
            side = max(w, h)
            square_x = x + (w - side) // 2
            square_y = y + (h - side) // 2

            # Draw square around the detected yellow object
            cv2.rectangle(frame, (square_x, square_y), (square_x + side, square_y + side), (0, 255, 0), 2)

    # Show the frame with square outlines
    cv2.imshow('Yellow Object Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
