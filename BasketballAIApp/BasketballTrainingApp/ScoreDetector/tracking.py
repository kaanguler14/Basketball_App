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
        # --- Oyuncu sayısını kullanıcıdan al ---
        print("\n" + "="*50)
        print("🏀 BASKETBOL TRACKING SISTEMI 🏀")
        print("="*50)
        while True:
            try:
                num_players = int(input("Kaç oyuncu var? (1 veya 2): "))
                if num_players in [1, 2]:
                    self.num_players = num_players
                    print(f"✓ {num_players} oyuncu seçildi!")
                    break
                else:
                    print("❌ Lütfen 1 veya 2 girin!")
            except ValueError:
                print("❌ Lütfen geçerli bir sayı girin!")
        print("="*50 + "\n")
        
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
        self.cap = cv2.VideoCapture(r"D:\repos\Basketball_App\BasketballAIApp\clips\training2.mp4")
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

        # --- MINIMAP TOGGLE ---
        self.show_minimap = True  # Minimap görünürlüğü (M tuşu ile toggle)
        
        # --- DETECTED PLAYERS ---
        self.detected_players = set()  # Tespit edilen oyuncu ID'leri

        # --- Homography ---
        ret, first_frame = self.cap.read()
        first_frame = cv2.resize(first_frame, (int(first_frame.shape[1]*scale), int(first_frame.shape[0]*scale)))
        self.video_points_dict, self.minimap_points_dict = ps.load_or_select(first_frame)
        self.H, self.use_flip, self.h_img = h.compute_homography(self.video_points_dict, self.minimap_points_dict)

        self.run()

    # ------------------------ RUN ------------------------
    def run(self):
        # Tam ekran pencere oluştur
        cv2.namedWindow("Basketball Tracker", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Basketball Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
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
                
                # Tespit edilen oyuncuları kaydet
                self.detected_players.add(track_id)

                cv2.circle(self.frame, (cx, cy), 5, (0, 255, 255), -1)
                cv2.putText(self.frame, f"P{track_id}", (int(l), int(t) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # --- BALL & HOOP CLEANING + SHOT DETECTION ---
            self.clean_motion()
            self.shot_detection(players)

            # --- MINIMAP --- (shot_detector modülünden shot_history)
            minimap_display = dm.draw_minimap(self.minimap_img,self.shot_detector.shot_history,players,self.H,self.use_flip,self.h_img)

            # --- MINIMAP OVERLAY (FIFA/PES tarzı) - M tuşu ile toggle ---
            if self.show_minimap:
                # Minimap boyutunu ayarla (frame'in %20'si kadar)
                minimap_scale = 0.2  # Minimap frame boyutunun %20'si
                minimap_width = int(self.frame.shape[1] * minimap_scale)
                minimap_height = int(minimap_width * (minimap_display.shape[0] / minimap_display.shape[1]))
                minimap_small = cv2.resize(minimap_display, (minimap_width, minimap_height))
                
                # Minimap pozisyonu (sağ alt köşe)
                margin = 20  # Kenardan boşluk
                y_offset = self.frame.shape[0] - minimap_height - margin
                x_offset = self.frame.shape[1] - minimap_width - margin
                
                # Minimap'i frame üzerine yerleştir (overlay)
                # Yarı saydam efekt için alpha blending
                alpha = 0.8  # Opaklık (0=tamamen saydam, 1=tamamen opak)
                overlay_region = self.frame[y_offset:y_offset+minimap_height, x_offset:x_offset+minimap_width]
                blended = cv2.addWeighted(overlay_region, 1-alpha, minimap_small, alpha, 0)
                self.frame[y_offset:y_offset+minimap_height, x_offset:x_offset+minimap_width] = blended
                
                # Minimap etrafına çerçeve çiz
                cv2.rectangle(self.frame, (x_offset-2, y_offset-2), 
                             (x_offset+minimap_width+2, y_offset+minimap_height+2), 
                             (255, 255, 255), 2)

            # --- SCORE OVERLAY --- (shot_detector modülünden)
            if self.shot_detector.fade_counter > 0:
                cv2.putText(self.frame, self.shot_detector.overlay_text, (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, self.shot_detector.overlay_color, 4)
            
            # --- SCOREBOARD --- (Dijital basketbol scoreboard)
            self.draw_scoreboard()

            cv2.putText(self.frame, f"FPS: {int(self.fps)}", (20, self.frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Minimap durumu göster (sol alt, FPS'in üstünde)
            minimap_status = "Minimap: ON (M)" if self.show_minimap else "Minimap: OFF (M)"
            cv2.putText(self.frame, minimap_status, (20, self.frame.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            self.frame_count += 1
            cv2.imshow("Basketball Tracker", self.frame)
            
            # Klavye kontrolü
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC tuşu - Çıkış
                break
            elif key == ord('m') or key == ord('M'):  # M tuşu - Minimap toggle
                self.show_minimap = not self.show_minimap
                status = "AÇIK" if self.show_minimap else "KAPALI"
                print(f"🗺️  Minimap: {status}")
        self.cap.release()
        cv2.destroyAllWindows()

    # ------------------------ SCOREBOARD ------------------------
    def draw_scoreboard(self):
        """Klasik basketbol scoreboard - dijital skor göstergesi"""
        player_scores = self.shot_detector.get_player_scores()
        
        # Scoreboard pozisyonu ve boyutu (oyuncu sayısına göre) - TOTAL olmadan
        if self.num_players == 1:
            board_width = 100
            board_height = 45
        else:  # 2 oyuncu
            board_width = 160
            board_height = 45
        
        start_x = 10
        start_y = 10
        
        # Overlay için
        overlay = self.frame.copy()
        
        # Ana scoreboard arka plan (koyu gri/siyah)
        cv2.rectangle(overlay, (start_x, start_y), 
                     (start_x + board_width, start_y + board_height), (25, 25, 25), -1)
        
        # Çerçeve (kalın beyaz)
        cv2.rectangle(overlay, (start_x, start_y), 
                     (start_x + board_width, start_y + board_height), (200, 200, 200), 2)
        
        # İç çerçeve (ince)
        cv2.rectangle(overlay, (start_x + 3, start_y + 3), 
                     (start_x + board_width - 3, start_y + board_height - 3), (100, 100, 100), 1)
        
        # Gerçek player skorlarını al
        all_players = {}
        
        # Tespit edilen oyuncuları kullan
        detected_list = sorted(list(self.detected_players))[:self.num_players]
        
        for player_id in detected_list:
            if player_id in player_scores:
                all_players[player_id] = player_scores[player_id]
            else:
                # Tespit edilmiş ama henüz şut atmamış
                all_players[player_id] = {"points": 0, "makes": 0, "attempts": 0}
        
        # Eğer henüz yeterli oyuncu tespit edilmediyse, placeholder ekle
        if len(detected_list) < self.num_players:
            for i in range(len(detected_list), self.num_players):
                placeholder_id = -(i + 1)
                all_players[placeholder_id] = {"points": 0, "makes": 0, "attempts": 0}
        
        # Oyuncuları puana göre sırala ve sadece seçilen sayıda göster
        sorted_players = sorted(all_players.items(), key=lambda x: x[1]['points'], reverse=True)[:self.num_players]
        
        # Oyuncu skorları (TOTAL olmadan, direkt başla)
        player_section_y = start_y + 8
        
        # Tüm oyuncuları göster
        player_width = board_width // self.num_players
        
        for idx, (player_id, stats) in enumerate(sorted_players):
            player_x = start_x + (idx * player_width)
            
            # Oyuncu bölmesi
            if idx > 0:
                # Dikey ayırıcı çizgi
                cv2.line(overlay, (player_x, start_y + 5), 
                        (player_x, start_y + board_height - 5), (60, 60, 60), 1)
            
            # Oyuncu etiketi (negatif ID'leri düzelt)
            try:
                if isinstance(player_id, int) and player_id < 0:
                    player_label = "---"  # Henüz tespit edilmemiş oyuncu
                else:
                    player_label = f"P{player_id}"
            except:
                player_label = f"P{player_id}"
            label_size = cv2.getTextSize(player_label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
            label_x = player_x + (player_width - label_size[0]) // 2
            cv2.putText(overlay, player_label, (label_x, player_section_y + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
            
            # Oyuncu puanı (büyük dijital)
            points = stats['points']
            points_text = f"{points:02d}"
            points_size = cv2.getTextSize(points_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            points_x = player_x + (player_width - points_size[0]) // 2
            
            # Renkli puan (sıralamaya göre)
            if idx == 0:
                points_color = (0, 215, 255)  # Altın
            elif idx == 1:
                points_color = (192, 192, 192)  # Gümüş
            else:
                points_color = (112, 162, 205)  # Bronz
            
            cv2.putText(overlay, points_text, (points_x, player_section_y + 28), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, points_color, 2)
            
            # İstatistikler (küçük)
            made_att = f"{stats['makes']}/{stats['attempts']}"
            accuracy = (stats['makes'] / stats['attempts'] * 100) if stats['attempts'] > 0 else 0
            stats_text = f"{accuracy:.0f}%"
            stats_size = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.25, 1)[0]
            stats_x = player_x + (player_width - stats_size[0]) // 2
            
            cv2.putText(overlay, stats_text, (stats_x, player_section_y + 38), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (150, 200, 255), 1)
        
        # Overlay'i uygula
        alpha = 0.90
        cv2.addWeighted(overlay, alpha, self.frame, 1 - alpha, 0, self.frame)

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
