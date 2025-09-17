# file: homography_overlay_final.py
import cv2
import numpy as np

# Template sahayı yükle
template = cv2.imread("D://BasketballSeg//testing//img_1.png")
h_temp, w_temp, _ = template.shape

# Template üzerindeki sabit noktalar (sen tıklayarak seçtin)
template_pts = np.array([
    [141, 175],  # pota altı
    [106,  89],  # sol köşe
    [175,  89],  # sağ köşe
    [141,  32]   # üç sayı üst
], dtype="float32")

# Video
cap = cv2.VideoCapture("D://BasketballSeg//clips//training3.mp4")
ret, first_frame = cap.read()
if not ret:
    print("Video açılamadı!")
    exit()

frame_pts = []

# Mouse callback
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(frame_pts) < 4:
            frame_pts.append([x, y])
            cv2.circle(first_frame, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Select 4 points", first_frame)
            print(f"Point {len(frame_pts)}: x={x}, y={y}")

# 1. Başlangıçta kullanıcıdan 4 nokta seçmesini iste
cv2.imshow("Select 4 points", first_frame)
cv2.setMouseCallback("Select 4 points", click_event)

while len(frame_pts) < 4:
    cv2.waitKey(1)

cv2.destroyWindow("Select 4 points")
frame_pts = np.array(frame_pts, dtype="float32")
print("Frame üzerindeki noktalar:", frame_pts)

# 2. Homography hesapla
H, _ = cv2.findHomography(template_pts, frame_pts)

# 3. Video boyunca overlay uygula
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # videoyu başa al
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Template'i warp et
    overlay = cv2.warpPerspective(template, H, (frame.shape[1], frame.shape[0]))
    result = cv2.addWeighted(frame, 1, overlay, 0.5, 0)

    cv2.imshow("Court Overlay", result)
    if cv2.waitKey(30) & 0xFF == 27:  # ESC ile çık
        break

cap.release()
cv2.destroyAllWindows()
