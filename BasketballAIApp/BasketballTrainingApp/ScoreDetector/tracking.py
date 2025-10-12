# shot_detector_deepsort.py

from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import pose
from utilsfixed import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device
import time
# ------------------ DeepSORT ------------------
from deep_sort_realtime.deepsort_tracker import DeepSort
import pointSelection as ps
from BasketballAIApp.BasketballTrainingApp.Homography import homography as h
import detection as det
import draw_minimap as dm
# ------------------ MODULAR SHOT DETECTOR ------------------
from shot_detector import ShotDetectorModule


# ------------------------ CONFIG ------------------------
court_labels = [
    "top_left", "top_right","top", "bottom_right", "bottom_left",
    "free_throw_left", "free_throw_right",
    "center_circle", "paint_left", "paint_right"
]

vp_json = "video_points.json"
mp_json = "minimap_points.json"
scale = 0.5  # video resize


class ShotDetector:
    def __init__(self):
        # --- YOLO MODELLERİ ---
        self.model_ball = YOLO("D://repos//Basketball_App//BasketballAIApp//Trainings//kagglebest.pt")
        self.model_player = YOLO("D://repos//Basketball_App//yolov8s.pt")  # person detection
        self.device = get_device()

        # --- DeepSORT Tracker ---
        self.deepsort = DeepSort(
            max_age=60,
            n_init=2,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3
        )

        # --- VIDEO / MINIMAP ---
        self.cap = cv2.VideoCapture(r"D:\repos\Basketball_App\BasketballAIApp\clips\training7.mp4")
        self.minimap_img = cv2.imread(r"D:\repos\Basketball_App\BasketballAIApp\BasketballTrainingApp\Homography\images\hom.png")
        self.frame_count = 0
        self.frame = None


        # --- BALL/HOOP LOGIC ---
        self.ball_pos = []
        self.hoop_pos = []
        
        # --- HOOP DETECTION OPTIMIZATION ---
        self.hoop_detected = False  # Pota tespit edildi mi?
        self.stable_hoop_pos = None  # Sabit pota pozisyonu (center, w, h)

        # --- MODULAR SHOT DETECTOR ---
        self.shot_detector = ShotDetectorModule()

        # --- FPS ---
        self.prev_time = 0.0
        self.fps = 0

        # --- Homography ---
        ret, first_frame = self.cap.read()
        first_frame = cv2.resize(first_frame, (int(first_frame.shape[1]*scale), int(first_frame.shape[0]*scale)))
        self.video_points_dict, self.minimap_points_dict = ps.load_or_select(first_frame)
        self.H, self.use_flip, self.h_img = h.compute_homography(self.video_points_dict, self.minimap_points_dict)

        self.run()

    # ------------------------ RUN ------------------------
    def run(self):
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break

            # --- FPS ---
            now = time.time()
            self.fps = 1.0 / (now - self.prev_time) if self.prev_time else 0.0
            self.prev_time = now

            # Pose Estimation

            #self.pose(self.frame)

            self.frame = cv2.resize(self.frame, (int(self.frame.shape[1] * scale), int(self.frame.shape[0] * scale)))


            #det(self.frame,self.hoop_detected,self.model_ball,self.model_player)

            # --- HOOP DETECTION (sadece bir kez) ---
            if not self.hoop_detected:
                results_ball = self.model_ball(self.frame, stream=True, device=self.device)
                for r in results_ball:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w, h = x2-x1, y2-y1
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        center = (x1 + w//2, y1 + h//2)
                        current_class = ["basketball", "rim"][cls]

                        if (conf>0.3 or (in_hoop_region(center,self.hoop_pos) and conf>0.15)) and current_class=="basketball":
                            self.ball_pos.append((center, self.frame_count, w, h, conf))
                            cvzone.cornerRect(self.frame, (x1, y1, w, h))
                        
                        if conf>0.5 and current_class=="rim":
                            self.hoop_pos.append((center, self.frame_count, w, h, conf))
                            cvzone.cornerRect(self.frame, (x1, y1, w, h))
                            
                            # Pota tespit edildi, sabit pozisyonu kaydet
                            if not self.hoop_detected and len(self.hoop_pos) >= 5:
                                # 5 frame'de tespit edildiyse, ortalamasını al ve sabitle
                                avg_cx = int(np.mean([pos[0][0] for pos in self.hoop_pos[-5:]]))
                                avg_cy = int(np.mean([pos[0][1] for pos in self.hoop_pos[-5:]]))
                                avg_w = int(np.mean([pos[2] for pos in self.hoop_pos[-5:]]))
                                avg_h = int(np.mean([pos[3] for pos in self.hoop_pos[-5:]]))
                                
                                self.stable_hoop_pos = ((avg_cx, avg_cy), self.frame_count, avg_w, avg_h, 1.0)
                                self.hoop_detected = True
                                print(f"✓ Pota tespit edildi ve sabitlend: {avg_cx}, {avg_cy} (w={avg_w}, h={avg_h})")
            else:
                # --- BALL DETECTION ONLY (pota zaten tespit edildi) ---
                results_ball = self.model_ball(self.frame, stream=True, device=self.device)
                for r in results_ball:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w, h = x2-x1, y2-y1
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        center = (x1 + w//2, y1 + h//2)
                        current_class = ["basketball", "rim"][cls]

                        # Sadece basketbol topunu tespit et
                        if (conf>0.3 or (in_hoop_region(center,self.hoop_pos) and conf>0.15)) and current_class=="basketball":
                            self.ball_pos.append((center, self.frame_count, w, h, conf))
                            cvzone.cornerRect(self.frame, (x1, y1, w, h))
                
                # Sabit pota pozisyonunu hoop_pos listesinde kullan
                if self.stable_hoop_pos and (len(self.hoop_pos) == 0 or self.hoop_pos[-1][1] < self.frame_count):
                    # Stable pozisyonu kullan
                    self.hoop_pos = [self.stable_hoop_pos]
                    
                # Potayı çiz (sabit pozisyon)
                hx, hy = self.stable_hoop_pos[0]
                hw, hh = self.stable_hoop_pos[2], self.stable_hoop_pos[3]
                cvzone.cornerRect(self.frame, (hx - hw//2, hy - hh//2, hw, hh), colorC=(0, 255, 0))

            # --- PLAYER DETECTION + DEEPSORT TRACKING ---
            results_player = self.model_player(self.frame, device=self.device, conf=0.6)[0]
            CONF_THRESHOLD = 0.65

            detections = []
            if results_player.boxes is not None:
                for box in results_player.boxes:
                    cls_index = int(box.cls.cpu().numpy())
                    cls_name = results_player.names[cls_index]
                    if cls_name != "person":
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    # --- sadece yeterli confidence varsa ekle ---
                    if conf >= CONF_THRESHOLD:
                        detections.append(((x1, y1, x2 - x1, y2 - y1), conf, cls_index))
            tracks = self.deepsort.update_tracks(detections, frame=self.frame)

            players = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                l, t, w, h = track.to_ltwh()
                cx, cy = int(l + w/2), int(t + h)
                players.append((cx, cy, track_id))

                cv2.circle(self.frame, (cx, cy), 5, (0, 255, 255), -1)
                cv2.putText(self.frame, f"P{track_id}", (int(l), int(t) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # --- BALL & HOOP CLEANING + SHOT DETECTION ---
            self.clean_motion()
            self.shot_detection(players)

            # --- MINIMAP --- (shot_detector modülünden shot_history)
            dm.draw_minimap(self.minimap_img,self.shot_detector.shot_history,players,self.H,self.use_flip,self.h_img)

            # --- SCORE OVERLAY --- (shot_detector modülünden)
            if self.shot_detector.fade_counter > 0:
                cv2.putText(self.frame, self.shot_detector.overlay_text, (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, self.shot_detector.overlay_color, 4)
            # --- TOTAL SCORE DISPLAY --- (shot_detector modülünden)
            score_text = f"Score: {self.shot_detector.makes}/{self.shot_detector.attempts}"
            cv2.putText(self.frame, score_text, (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)


            cv2.putText(self.frame, f"FPS: {int(self.fps)}", (20, self.frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            self.frame_count += 1
            cv2.imshow("Frame", self.frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        self.cap.release()
        cv2.destroyAllWindows()

    # ------------------------ CLEAN / DETECT ------------------------
    def clean_motion(self):
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for b in self.ball_pos:
            cv2.circle(self.frame, b[0], 2, (255,0,255), 2)
            if len(self.hoop_pos) > 1:
                self.hoop_pos = clean_hoop_pos(self.hoop_pos)
                cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (0, 128, 0), 2)

    def shot_detection(self, players=[]):

        if len(self.ball_pos) == 0:
            return
        
        # Release point detection (modüler)
        release_info = self.shot_detector.detect_shot(
            self.ball_pos, self.hoop_pos, players, 
            self.frame, self.frame_count
        )
        
        # Shot scoring (modüler)
        shot_data = self.shot_detector.score_shot(
            self.ball_pos, self.hoop_pos,
            self.H, self.use_flip, self.h_img,
            players
        )
        
        # Fade counter'ı güncelle
        self.shot_detector.update_fade()
        
        return release_info, shot_data


if __name__ == "__main__":
    ShotDetector()
