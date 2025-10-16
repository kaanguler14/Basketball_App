from ultralytics import YOLO
import cv2
import time

# Model yükle
model = YOLO("D://repos//Basketball_App//BasketballAIApp//Trainings//kagglebest.pt")

video_path = "D://repos//Basketball_App//BasketballAIApp//clips//training2.mp4"
cap = cv2.VideoCapture(video_path)

prev_time = 0
score = 0
prev_ty = None
scored_this_frame = False  # Tek seferlik skor kontrolü

# Otomatik class ID tespiti için
ball_class = None
rim_class = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))

    # Model tahmini
    results = model.predict(frame, conf=0.6)

    top_center = None
    rim_box = None

    for det in results[0].boxes:
        cls = int(det.cls[0])
        x1, y1, x2, y2 = map(int, det.xyxy[0])

        # İlk frame’de class ID belirleme
        if ball_class is None and "ball" in model.names[cls].lower():
            ball_class = cls
        if rim_class is None and "rim" in model.names[cls].lower():
            rim_class = cls

        # Top
        if cls == ball_class:
            top_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(frame, top_center, 5, (0, 0, 255), -1)
        # Rim
        elif cls == rim_class:
            rim_box = (x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Skor kontrolü
    if top_center and rim_box:
        tx, ty = top_center
        x1, y1, x2, y2 = rim_box
        margin = 5

        # Top aşağı hareket ediyor ve daha önce skor sayılmadıysa
        if prev_ty is not None and ty > prev_ty and not scored_this_frame:
            if x1 - margin <= tx <= x2 + margin and y1 - margin <= ty <= y2 + margin:
                score += 1
                scored_this_frame = True  # Tek seferlik skor
                print("SCORE!", score)
        prev_ty = ty
    else:
        scored_this_frame = False  # Top veya rim yoksa sonraki frame’de tekrar sayabilir

    # FPS hesaplama
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"SCORE: {score}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Görüntüyü göster
    cv2.imshow("Basketball Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
