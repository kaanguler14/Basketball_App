# shot_detector_deepsort.py

from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import json
import os
from utilsfixed import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device
import time
# ------------------ DeepSORT ------------------
from deep_sort_realtime.deepsort_tracker import DeepSort
import tkinter as tk
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk
import mediapipe as mp
import pointSelection as ps

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

        #MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic=mp.solutions.holistic
        self.holistic=self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # --- BALL/HOOP LOGIC ---
        self.ball_pos = []
        self.hoop_pos = []
        self.makes = 0
        self.attempts = 0
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0
        
        # --- HOOP DETECTION OPTIMIZATION ---
        self.hoop_detected = False  # Pota tespit edildi mi?
        self.stable_hoop_pos = None  # Sabit pota pozisyonu (center, w, h)

        #Shot location & Release Point Detection
        self.shot_history = []
        self.ball_with_player = False  # Top oyuncuda mı?
        self.release_detected = False  # Şut atıldı mı?
        self.release_frame = None  # Şutun atıldığı frame
        self.release_player_pos = None  # Şut anındaki oyuncu pozisyonu
        self.shooter_id = None  # Şut atan oyuncu ID'si
        self.ball_player_history = []  # Top-oyuncu mesafe geçmişi




        # --- FPS ---
        self.prev_time = 0.0
        self.fps = 0

        # --- OVERLAY ---
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_text = "waiting.."
        self.overlay_color = (0, 0, 0)

        # --- Homography ---
        ret, first_frame = self.cap.read()
        first_frame = cv2.resize(first_frame, (int(first_frame.shape[1]*scale), int(first_frame.shape[0]*scale)))
        self.video_points_dict, self.minimap_points_dict = ps.load_or_select(first_frame)
        self.H, self.use_flip, self.h_img = self.compute_homography()

        self.shot_history = []  # x,y,made
        self.run()

    # ------------------------ HOMOGRAPHY ------------------------


    def compute_homography(self):
        common_labels = [l for l in court_labels if l in self.video_points_dict and l in self.minimap_points_dict]
        if len(common_labels) < 4:
            raise ValueError("Homography için en az 4 ortak nokta gerekli!")
        video_pts = np.array([self.video_points_dict[l] for l in common_labels], dtype=np.float32)
        minimap_pts = np.array([self.minimap_points_dict[l] for l in common_labels], dtype=np.float32)

        H1, err1, proj1, errs1 = self.compute_h_and_error(video_pts, minimap_pts)
        h_img = self.minimap_img.shape[0]
        minimap_pts_flipped = np.array([[x, h_img - y] for (x, y) in minimap_pts], dtype=np.float32)
        H2, err2, proj2, errs2 = self.compute_h_and_error(video_pts, minimap_pts_flipped)

        use_flip = False
        H, proj, errs, err = H1, proj1, errs1, err1
        if H1 is None and H2 is None:
            raise ValueError("Homography hesaplanamadı")
        elif H1 is None or (err2 + 1.0 < err1):
            use_flip = True
            H, proj, errs, err = H2, proj2, errs2, err2
        print(f"Seçilen dönüşüm: flip_y={use_flip}, ortalama reproj hatası={err:.2f}")
        return H, use_flip, h_img

    def compute_h_and_error(self, video_pts, minimap_pts):
        H, mask = cv2.findHomography(video_pts, minimap_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None, float('inf'), None, None
        proj = cv2.perspectiveTransform(video_pts.reshape(-1,1,2), H).reshape(-1,2)
        errs = np.linalg.norm(proj - minimap_pts, axis=1)
        return H, float(np.mean(errs)), proj, errs

    #Pose Estimation
    def pose(self,frame):
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        poseResults = self.holistic.process(self.frame)
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)

        self.mp_drawing.draw_landmarks(self.frame, poseResults.face_landmarks,
                                       self.mp_holistic.FACEMESH_CONTOURS,
                                       self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1,
                                                                   circle_radius=1),
                                       self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1,
                                                                   circle_radius=1))

        self.mp_drawing.draw_landmarks(self.frame, poseResults.right_hand_landmarks,
                                       self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2,
                                                                   circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2,
                                                                   circle_radius=2))

        self.mp_drawing.draw_landmarks(self.frame, poseResults.left_hand_landmarks,
                                       self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2,
                                                                   circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2,
                                                                   circle_radius=2))

        self.mp_drawing.draw_landmarks(self.frame, poseResults.pose_landmarks,
                                       self.mp_holistic.POSE_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                   circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2,
                                                                   circle_radius=2))

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

            # --- MINIMAP ---
            self.draw_minimap(players)

            # --- SCORE OVERLAY ---
            if self.fade_counter > 0:
                cv2.putText(self.frame, self.overlay_text, (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, self.overlay_color, 4)
            # --- TOTAL SCORE DISPLAY ---
            score_text = f"Score: {self.makes}/{self.attempts}"
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
        """
        Gelişmiş şut tespiti - şutun elden çıktığı anı tespit eder
        """
        if len(self.ball_pos) == 0:
            return
            
        # --- RELEASE POINT DETECTION (Gelişmiş - Uzaktan şutlar için optimize) ---
        if players and len(self.ball_pos) > 0:
            bx, by = self.ball_pos[-1][0]
            
            # En yakın oyuncuyu bul
            nearest = min(players, key=lambda p: math.sqrt((p[0] - bx) ** 2 + (p[1] - by) ** 2))
            nearest_dist = math.sqrt((nearest[0] - bx) ** 2 + (nearest[1] - by) ** 2)
            px, py = nearest[0], nearest[1]
            
            # Top-oyuncu mesafe geçmişini kaydet
            self.ball_player_history.append({
                'frame': self.frame_count,
                'distance': nearest_dist,
                'player_pos': (px, py),
                'player_id': nearest[2] if len(nearest) > 2 else None,
                'ball_pos': (bx, by)
            })
            
            # Son 15 frame'i tut (uzaktan şutlar için daha uzun geçmiş)
            if len(self.ball_player_history) > 15:
                self.ball_player_history.pop(0)
            
            # Potaya olan mesafeyi hesapla (dinamik threshold için)
            if len(self.hoop_pos) > 0:
                hx, hy = self.hoop_pos[-1][0]
                player_to_hoop_dist = math.sqrt((px - hx) ** 2 + (py - hy) ** 2)
            else:
                player_to_hoop_dist = 200  # default
            
            # PERSPEKTİF KOMPANSASYONU - Y koordinatına göre (derinlik)
            # Y büyük = kameraya YAKIN (alt kısım, önde) → daha büyük threshold gerekli
            # Y küçük = kameraya UZAK (üst kısım, arkada) → daha küçük threshold
            frame_height = self.frame.shape[0]
            
            # Y pozisyonuna göre perspektif faktörü (0.8 - 1.5 arası)
            # Alt kısım (py/height > 0.7): faktör ~1.5 (threshold %50 artar)
            # Orta kısım (py/height ~ 0.5): faktör ~1.0 (normal)
            # Üst kısım (py/height < 0.3): faktör ~0.8 (threshold %20 azalır)
            y_ratio = py / frame_height
            if y_ratio > 0.7:  # Çok önde (kameraya çok yakın)
                perspective_factor = 1.5
                depth_zone = "ÖN"
            elif y_ratio > 0.5:  # Orta-ön
                perspective_factor = 1.2
                depth_zone = "ORTA-ÖN"
            elif y_ratio > 0.3:  # Orta-arka
                perspective_factor = 1.0
                depth_zone = "ORTA"
            else:  # Çok arkada
                perspective_factor = 0.8
                depth_zone = "ARKA"
            
            # DİNAMİK THRESHOLD - Potaya uzaklığa göre ayarlanır
            # Uzak şut: daha büyük threshold
            # Yakın şut: daha küçük threshold
            BASE_THRESHOLD = 50
            if player_to_hoop_dist > 300:  # Çok uzak şut (3-point)
                HOLDING_THRESHOLD = 80
                RELEASE_THRESHOLD = 15  # Daha düşük, çünkü hızlı hareket var
            elif player_to_hoop_dist > 200:  # Orta mesafe
                HOLDING_THRESHOLD = 65
                RELEASE_THRESHOLD = 18
            else:  # Yakın şut
                HOLDING_THRESHOLD = 50
                RELEASE_THRESHOLD = 20
            
            # Perspektif faktörünü uygula
            HOLDING_THRESHOLD = int(HOLDING_THRESHOLD * perspective_factor)
            RELEASE_THRESHOLD = int(RELEASE_THRESHOLD / perspective_factor)  # Ters oran (daha hassas tespit)
            
            # Top oyuncuya yakınsa = oyuncuda
            if nearest_dist < HOLDING_THRESHOLD:
                # İlk kez topu tutuyorsa shooter ID'yi kaydet
                if not self.ball_with_player:
                    self.shooter_id = nearest[2] if len(nearest) > 2 else None
                    print(f"   → Top yakalandı: Oyuncu P{self.shooter_id}")
                
                self.ball_with_player = True
                self.release_detected = False
            
            # Top oyuncudan uzaklaşıyorsa = ŞUT ATILDI!
            elif self.ball_with_player and not self.release_detected:
                # En az 2 frame gerekli
                if len(self.ball_player_history) >= 2:
                    # Son frame'leri al
                    recent = self.ball_player_history[-2:]
                    
                    # Mesafe artışı
                    dist_increase = recent[-1]['distance'] - recent[0]['distance']
                    
                    # Y hareketi (yukarı = negatif, çünkü koordinat sistemi)
                    y_movement = recent[-1]['ball_pos'][1] - recent[0]['ball_pos'][1]
                    
                    # Hız hesapla (piksel/frame)
                    ball_velocity = math.sqrt(
                        (recent[-1]['ball_pos'][0] - recent[0]['ball_pos'][0]) ** 2 +
                        (recent[-1]['ball_pos'][1] - recent[0]['ball_pos'][1]) ** 2
                    )
                    
                    # RELEASE KRİTERLERİ:
                    # 1. Mesafe artıyor VEYA
                    # 2. Top yukarı hareket ediyor (y_movement < 0) VE hızlı hareket var VEYA
                    # 3. Velocity yüksek (hızlı atış)
                    
                    release_condition = (
                        dist_increase > RELEASE_THRESHOLD or  # Mesafe artışı
                        (y_movement < -10 and ball_velocity > 15) or  # Yukarı + hızlı
                        ball_velocity > 25  # Çok hızlı hareket (ani atış)
                    )
                    
                    if release_condition:
                        self.release_detected = True
                        self.release_frame = self.frame_count
                        
                        # ÖNEMLİ: Şutu atan oyuncunun pozisyonunu bul
                        # Sadece AYNI OYUNCUYA ait frame'leri kullan (shooter_id ile eşleşenler)
                        best_idx = -1
                        shooter_frames = []
                        
                        # Geriye doğru git ve shooter_id'ye sahip frame'leri bul
                        for i in range(len(self.ball_player_history) - 1, -1, -1):
                            frame_data = self.ball_player_history[i]
                            # Aynı oyuncu MU? (shooter_id ile eşleşiyor mu?)
                            if frame_data['player_id'] == self.shooter_id:
                                # Top bu oyuncuya yakın mıydı?
                                if frame_data['distance'] < HOLDING_THRESHOLD * 1.3:
                                    shooter_frames.append(i)
                                    if best_idx == -1:
                                        best_idx = i
                            
                            # Son 8 frame'e bak (yeterli)
                            if len(shooter_frames) >= 5:
                                break
                        
                        # Eğer shooter'a ait frame bulunamazsa (ID kaydı yoksa), en yakın olanı al
                        if best_idx == -1:
                            for i in range(len(self.ball_player_history) - 1, max(0, len(self.ball_player_history) - 4), -1):
                                if self.ball_player_history[i]['distance'] < HOLDING_THRESHOLD * 1.2:
                                    best_idx = i
                                    break
                        
                        # En iyi indeksi kullan
                        if best_idx != -1:
                            self.release_player_pos = self.ball_player_history[best_idx]['player_pos']
                            # Shooter ID'yi de doğrula (emin olmak için)
                            confirmed_shooter_id = self.ball_player_history[best_idx]['player_id']
                            if confirmed_shooter_id is not None:
                                self.shooter_id = confirmed_shooter_id
                        else:
                            # Fallback: en son pozisyon
                            self.release_player_pos = self.ball_player_history[-1]['player_pos']
                        
                        # Detaylı log
                        shot_type = "UZAK" if player_to_hoop_dist > 300 else ("ORTA" if player_to_hoop_dist > 200 else "YAKIN")
                        print(f"🏀 {shot_type} ŞUT ATILDI! [Derinlik: {depth_zone}]")
                        print(f"   Frame: {self.frame_count}, Oyuncu: P{self.shooter_id} ({'✓ Doğrulandı' if len(shooter_frames) > 0 else '⚠ Fallback'})")
                        print(f"   Pozisyon: {self.release_player_pos}, Y-oranı: {y_ratio:.2f}")
                        print(f"   Mesafe artışı: {dist_increase:.1f}px, Hız: {ball_velocity:.1f}px/f")
                        print(f"   Potaya mesafe: {player_to_hoop_dist:.0f}px")
                        print(f"   Threshold: HOLDING={HOLDING_THRESHOLD}px, RELEASE={RELEASE_THRESHOLD}px (faktör={perspective_factor:.1f})")
                        print(f"   Shooter frame sayısı: {len(shooter_frames)} (aynı oyuncuya ait)")
                        
                        # Frame'de işaretle
                        if self.release_player_pos:
                            cv2.circle(self.frame, self.release_player_pos, 15, (255, 0, 255), 3)
                            # Oyuncu ID'sini büyük göster
                            cv2.putText(self.frame, f"P{self.shooter_id} RELEASE ({shot_type})", 
                                      (self.release_player_pos[0] - 80, self.release_player_pos[1] - 35),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                            # Zone ve threshold bilgisi
                            cv2.putText(self.frame, f"{depth_zone} | H:{HOLDING_THRESHOLD} R:{RELEASE_THRESHOLD}", 
                                      (self.release_player_pos[0] - 80, self.release_player_pos[1] - 15),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                            # Doğrulama işareti
                            if len(shooter_frames) > 0:
                                cv2.putText(self.frame, "VERIFIED", 
                                          (self.release_player_pos[0] - 30, self.release_player_pos[1] + 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # --- SHOT SCORING (orijinal mantık) ---
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]

            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]

            if self.up and self.down and self.up_frame < self.down_frame:
                self.attempts += 1
                made = score(self.ball_pos, self.hoop_pos)
                if made:
                    self.makes += 1
                    self.overlay_text = "Score"
                    self.overlay_color = (0, 255, 0)
                else:
                    self.overlay_text = "Miss"
                    self.overlay_color = (0, 0, 255)
                self.fade_counter = self.fade_frames

                # --- Minimapte RELEASE POINT'i işaretle ---
                if self.release_player_pos:
                    # Release pozisyonunu kullan (şutun atıldığı yer)
                    pt = np.array([[[self.release_player_pos[0], self.release_player_pos[1]]]], dtype=np.float32)
                    proj_pt = cv2.perspectiveTransform(pt, self.H)[0][0]
                    mx = int(proj_pt[0])
                    my = int(proj_pt[1])
                    if self.use_flip:
                        my = self.h_img - my
                    self.shot_history.append((mx, my, made, self.shooter_id))
                    print(f"📍 Minimap'e eklendi: ({mx}, {my}), Oyuncu: P{self.shooter_id}, {'BAŞARILI' if made else 'BAŞARISIZ'}")
                else:
                    # Fallback: şu anki oyuncu pozisyonu
                    if players:
                        bx, by = self.ball_pos[-1][0]
                        nearest = min(players, key=lambda p: (p[0] - bx) ** 2 + (p[1] - by) ** 2)
                        pt = np.array([[[nearest[0], nearest[1]]]], dtype=np.float32)
                        proj_pt = cv2.perspectiveTransform(pt, self.H)[0][0]
                        mx = int(proj_pt[0])
                        my = int(proj_pt[1])
                        if self.use_flip:
                            my = self.h_img - my
                        pid = nearest[2] if len(nearest) > 2 else None
                        self.shot_history.append((mx, my, made, pid))

                # Reset
                self.up = False
                self.down = False
                self.ball_with_player = False
                self.release_detected = False
                self.release_player_pos = None
                self.shooter_id = None
                self.ball_player_history = []

        if self.fade_counter > 0:
            self.fade_counter -= 1

    def get_nearest_player(self, ball_pos, players):
        # ball_pos = (x,y)
        if not players:
            return (ball_pos[0], ball_pos[1])
        bx, by = ball_pos
        nearest = min(players, key=lambda p: (p[0] - bx) ** 2 + (p[1] - by) ** 2)
        return (nearest[0], nearest[1])

    def draw_minimap(self, players=[]):
        minimap_copy = self.minimap_img.copy()

        # Şut pozisyonlarını işaretle (RELEASE POINT'ler)
        for shot_data in self.shot_history:
            if len(shot_data) == 4:
                mx, my, made, shooter_id = shot_data
            else:
                mx, my, made = shot_data
                shooter_id = None
            
            # Renk: Yeşil = başarılı, Kırmızı = kaçan
            color = (0, 255, 0) if made else (0, 0, 255)
            
            # X işareti (şut pozisyonu)
            cv2.line(minimap_copy, (mx - 10, my - 10), (mx + 10, my + 10), color, 3)
            cv2.line(minimap_copy, (mx - 10, my + 10), (mx + 10, my - 10), color, 3)
            
            # Daire (release point vurgusu)
            cv2.circle(minimap_copy, (mx, my), 8, color, 2)
            
            # Oyuncu ID'si (varsa)
            if shooter_id is not None:
                cv2.putText(minimap_copy, f"P{shooter_id}", (mx + 12, my - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Oyuncular (cx,cy,ID)
        for cx, cy, pid in players:
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            proj_pt = cv2.perspectiveTransform(pt, self.H)[0][0]
            display_x = int(proj_pt[0])
            display_y = int(proj_pt[1])
            if self.use_flip:
                display_y = self.h_img - display_y
            inside = (0 <= display_x < minimap_copy.shape[1] and 0 <= display_y < minimap_copy.shape[0])
            color = (0, 0, 255) if inside else (0, 0, 128)
            cv2.circle(minimap_copy, (np.clip(display_x, 0, minimap_copy.shape[1] - 1),
                                      np.clip(display_y, 0, minimap_copy.shape[0] - 1)), 8, color, -1)
            cv2.putText(minimap_copy, f"P{pid}", (display_x + 8, display_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            if not inside:
                cv2.putText(minimap_copy, "OUT", (np.clip(display_x, 0, minimap_copy.shape[1] - 1) + 6,
                                                  np.clip(display_y, 0, minimap_copy.shape[0] - 1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Minimap AR", minimap_copy)


if __name__ == "__main__":
    ShotDetector()
