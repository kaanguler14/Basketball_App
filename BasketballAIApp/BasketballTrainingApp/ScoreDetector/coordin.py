# file: select_template_points.py
import cv2
import numpy as np

# Template sahayı yükle
template = cv2.imread("D://BasketballSeg//testing//img_1.png")
template_pts = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        template_pts.append([x, y])
        cv2.circle(template, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Template Points", template)
        print(f"Point {len(template_pts)}: x={x}, y={y}")

cv2.imshow("Select Template Points", template)
cv2.setMouseCallback("Select Template Points", click_event)

print("Template üzerinde 4 noktayı sırayla tıklayın: pota altı, sol köşe, sağ köşe, üç sayı üst")

while len(template_pts) < 4:
    cv2.waitKey(1)

cv2.destroyAllWindows()

template_pts = np.array(template_pts, dtype="float32")
print("Seçilen template noktaları:", template_pts)
