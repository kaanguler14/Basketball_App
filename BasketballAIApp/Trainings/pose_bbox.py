import cv2
import mediapipe as mp
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import time

# --- Ayarlar ---
VIDEO_SRC = r"D:\repos\Basketball_App\BasketballAIApp\clips\training7.mp4"
MIN_CONFIDENCE = 0.35
BBOX_EXPAND = 0.15
DISPLAY_SCALE = 0.6  # ekrana küçük sığması için

# --- MediaPipe Pose ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

# --- DeepSort ---
tracker = DeepSort(max_age=30,
                   n_init=3,
                   nms_max_overlap=1.0,
                   max_cosine_distance=0.2,
                   embedder="mobilenet")

def landmarks_to_bbox(landmarks, img_w, img_h, expand_ratio=BBOX_EXPAND):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    if not xs or not ys:
        return None
    x_min = min(xs) * img_w
    x_max = max(xs) * img_w
    y_min = min(ys) * img_h
    y_max = max(ys) * img_h

    w = x_max - x_min
    h = y_max - y_min

    # Genişlet
    x_min -= w * expand_ratio
    y_min -= h * expand_ratio
    x_max += w * expand_ratio
    y_max += h * expand_ratio

    # Ekran sınırlarını aşmasın
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_w - 1, x_max)
    y_max = min(img_h - 1, y_max)

    return int(x_min), int(y_min), int(x_max), int(y_max)


def main():
    cap = cv2.VideoCapture(VIDEO_SRC)
    cv2.namedWindow("DeepSort + MediaPipe Pose", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("DeepSort + MediaPipe Pose", 960, 540)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_h, img_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        detections = []

        # --- Pose çıkarımı ---
        if results.pose_landmarks:
            lm_list = results.pose_landmarks.landmark
            bbox = landmarks_to_bbox(lm_list, img_w, img_h)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                conf = 0.9
                detections.append(([x1, y1, x2, y2], conf, "person"))

        # --- DeepSort Takip ---
        tracks = tracker.update_tracks(detections, frame=frame)

        # --- Çizimler ---
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            l, t, r, b = map(int, ltrb)

            # Bounding box + ID
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # --- Pose çizimi (iskelet) ---
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_style.get_default_pose_landmarks_style()
            )

        # FPS hesapla
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Ekrana sığdır
        frame_display = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        cv2.imshow("DeepSort + MediaPipe Pose", frame_display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
