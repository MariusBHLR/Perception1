import cv2 as cv
import numpy as np

tol = 50 #tolérance 
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))

img = cv.imread("C:\\Users\\Marie-Pierre\\prog\\perception\\balle3.jpg") # Photo balle
if img is None:
    print("error")
    exit()

### Traitement (OPEN/CLOSE)  #####

img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
hsv = cv.bilateralFilter(hsv, d=9, sigmaColor=75, sigmaSpace=75)

###### Détection couleur/teinte (RGB/HSV) #############

def fct(event, x, y, flags, param):
    global img, output

    if event == cv.EVENT_RBUTTONDOWN:
        pixel = img[y, x]

        b0, g0, r0 = int(pixel[0]), int(pixel[1]), int(pixel[2])

        print(f"[RGB] Clic en ({x},{y}) - BGR = ({b0},{g0},{r0})")

        B = img[:, :, 0].astype(int)
        G = img[:, :, 1].astype(int)
        R = img[:, :, 2].astype(int)

        dist = np.sqrt((B - b0)**2 + (G - g0)**2 + (R - r0)**2)
        mask = (dist < tol).astype(np.uint8) * 255

        output = img.copy()
        output[mask > 0] = [255, 0, 0] #colorie en bleu

    elif event == cv.EVENT_LBUTTONDOWN:
        pixel = hsv[y, x]
        h0, s0, v0 = int(pixel[0]), int(pixel[1]), int(pixel[2])

        print(f"[HSV] Clic en ({x},{y}) - HS = ({h0},{s0}, {v0})")

        H = hsv[:, :, 0].astype(int)
        S = hsv[:, :, 1].astype(int)
        V = hsv[:, :, 2].astype(int)
        
        dh = np.minimum(np.abs(H - h0), 180 - np.abs(H - h0))  # distance circulaire
        dist = np.sqrt(dh**2 + (S-s0)**2)
        mask = (dist < tol).astype(np.uint8) * 255

        output = img.copy()
        output[mask > 0] = [0, 255, 0] #VERT pour HSV


##### Détection cercle #############

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_blur = cv.GaussianBlur(img_gray, (9, 9), 2)
cimg = cv.cvtColor(img_blur, cv.COLOR_GRAY2BGR)

circle = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, 1.2, 100, param1 = 100, param2 = 50, minRadius = 0, maxRadius = 0)



if circle is not None:
    circle = np.uint16(np.around(circle))
    # Affichage cercle/centre
    x, y, r = circle[0, 0]
    cv.circle(img, (x, y), r, (0, 255, 0), 3)
    cv.circle(img, (x, y), 2, (0, 0, 255), 3)

    
output = img.copy()


# Associer la fonction souris
cv.namedWindow("Selection")
cv.setMouseCallback("Selection", fct)


while True:

    cv.imshow("Selection", output)
    #cv.imshow("cercle", cimg)

    key = cv.waitKey(20) & 0xFF
    if key == 27: #ESC pour quitter
        break


cv.destroyAllWindows()

