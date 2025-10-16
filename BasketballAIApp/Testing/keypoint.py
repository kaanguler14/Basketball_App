from ultralytics import YOLO
import cv2
import torch

# --- Model ---
model_path = r"/BasketballAIApp/Models/best.pt"
model = YOLO(model_path)

# --- Video ---
video_path = r"D:\repos\Basketball_App\BasketballAIApp\clips\training2.mp4"
cap = cv2.VideoCapture(video_path)

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Inference ---
    results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)

    for r in results:
        if hasattr(r, 'keypoints') and r.keypoints is not None:
            for obj_kpts in r.keypoints:
                # Tensor veya liste olabilir
                if isinstance(obj_kpts, torch.Tensor):
                    obj_kpts_list = obj_kpts.detach().cpu().tolist()
                else:
                    obj_kpts_list = obj_kpts  # zaten liste

                # Her keypoint'i tek tek çiz
                for kp in obj_kpts_list:
                    if isinstance(kp, (list, tuple)) and len(kp) >= 3:
                        x, y, v = kp[:3]
                        if v > 0:
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    cv2.imshow("Keypoint Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC ile çıkış
        break

    frame_id += 1
    if frame_id % 50 == 0:
        print(f"Processed frame {frame_id}")

cap.release()
cv2.destroyAllWindows()
print("Realtime inference tamamlandı.")
