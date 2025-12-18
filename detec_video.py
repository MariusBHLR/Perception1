import cv2 as cv
import numpy as np


video_path = "C:\\Users\\Marie-Pierre\\prog\\perception\\balle.mp4"
cap = cv.VideoCapture(0)  # 0 to use webcam / video_path for a local file
if not cap.isOpened():
    print("error")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
 
    # Verification frame
    if not ret:
        print("frame error")
        break
    # Treatement image
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(gray, (9, 9), 2)
    #cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    bis = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    cimg = cv.morphologyEx(bis, cv.MORPH_CLOSE, kernel)

    circle = cv.HoughCircles(cimg, cv.HOUGH_GRADIENT, 1.2, 100, param1 = 100, param2 = 50, minRadius = 0, maxRadius = 0)

    if circle is not None:
        circle = np.uint16(np.around(circle))
        # Show circle/center
        x, y, r = circle[0, 0]
        cv.circle(frame, (x, y), r, (0, 255, 0), 3)
        cv.circle(frame, (x, y), 2, (0, 0, 255), 3)

    #resultat
    cv.imshow('frame', frame)
    key = cv.waitKey(20) & 0xFF
    if key == 27: #ESC to leave
        break
 

cap.release()
cv.destroyAllWindows()

