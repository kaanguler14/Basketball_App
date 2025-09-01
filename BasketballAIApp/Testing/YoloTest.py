from ultralytics import YOLO
import cv2
import time

# Model yükle
model = YOLO("D://BasketballAIApp//Trainings//runs//detect//train2//weights//best.pt")

# Video aç
video_path = "D://BasketballAIApp//clips//trainingclips.mp4"
cap = cv2.VideoCapture(video_path)

# FPS ölçümü için başlangıç
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Çözünürlüğü küçült (ör: yarı boyut)
    frame = cv2.resize(frame, (640, 360))

    # Model çalıştır
    results = model.track(frame, persist=True, classes=[0], conf=0.6)
    frame_ = results[0].plot()

    # FPS hesapla
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # FPS yazdır
    cv2.putText(frame_, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Görüntüyü göster
    cv2.imshow("Tracking", frame_)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
