# shot_detector_deepsort.py

from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont



from utilsfixed import in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device
from deep_sort_realtime.deepsort_tracker import DeepSort
import pointSelection as ps
from BasketballAIApp.BasketballTrainingApp.ScoreDetector import homography as h
import draw_minimap as dm
from shot_detector import ShotDetectorModule

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
        self.model_player = YOLO(r"D:\repos\Basketball_App\BasketballAIApp\Models\yolov8s.pt")  # person detection
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
        self.minimap_img = cv2.imread(r"D:\repos\Basketball_App\BasketballAIApp\BasketballTrainingApp\images\hom.png")
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

    # ------------------------ SCOREBOARD (PIL) ------------------------
    def draw_scoreboard(self):
        """Modern estetik scoreboard - PIL ile"""
        player_scores = self.shot_detector.get_player_scores()
        
        # Scoreboard boyutu (biraz daha büyük)
        if self.num_players == 1:
            board_width = 140
            board_height = 65
        else:  # 2 oyuncu
            board_width = 240
            board_height = 65
        
        # PIL Image oluştur (RGBA for transparency)
        scoreboard_img = Image.new('RGBA', (board_width, board_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(scoreboard_img)
        
        # Gradient arka plan oluştur (daha belirgin)
        for y in range(board_height):
            # Üstten alta: koyu mavi -> siyah gradient
            progress = y / board_height
            r = int(20 - progress * 10)
            g = int(25 - progress * 15)
            b = int(40 - progress * 20)
            alpha = int(240 - progress * 30)
            color = (r, g, b, alpha)
            draw.rectangle([(0, y), (board_width, y + 1)], fill=color)
        
        # Çerçeve (dış - parlak)
        draw.rectangle([(0, 0), (board_width - 1, board_height - 1)], 
                      outline=(255, 255, 255, 255), width=2)
        # İç çerçeve (ince - subtle)
        draw.rectangle([(3, 3), (board_width - 4, board_height - 4)], 
                      outline=(100, 100, 120, 200), width=1)
        
        # Font yükle (system fonts kullan)
        try:
            # Windows için tam yol
            font_path = r"C:\Windows\Fonts\arial.ttf"
            font_path_bold = r"C:\Windows\Fonts\arialbd.ttf"
            font_label = ImageFont.truetype(font_path, 13)
            font_score = ImageFont.truetype(font_path_bold, 26)  # Bold
            font_stats = ImageFont.truetype(font_path, 11)
            print("✓ Fonts loaded successfully")
        except Exception as e:
            print(f"⚠️ Font loading failed: {e}, using default")
            # Fallback - daha büyük boyutlar
            font_label = ImageFont.load_default()
            font_score = ImageFont.load_default()
            font_stats = ImageFont.load_default()
        
        # Gerçek player skorlarını al
        all_players = {}
        detected_list = sorted(list(self.detected_players))[:self.num_players]
        
        for player_id in detected_list:
            if player_id in player_scores:
                all_players[player_id] = player_scores[player_id]
            else:
                all_players[player_id] = {"points": 0, "makes": 0, "attempts": 0}
        
        if len(detected_list) < self.num_players:
            for i in range(len(detected_list), self.num_players):
                placeholder_id = -(i + 1)
                all_players[placeholder_id] = {"points": 0, "makes": 0, "attempts": 0}
        
        sorted_players = sorted(all_players.items(), key=lambda x: x[1]['points'], reverse=True)[:self.num_players]
        
        # Oyuncuları çiz
        player_width = board_width // self.num_players
        
        for idx, (player_id, stats) in enumerate(sorted_players):
            player_x = idx * player_width
            
            # Dikey ayırıcı (2. oyuncudan itibaren) - daha belirgin
            if idx > 0:
                # Gölge
                draw.line([(player_x + 1, 5), (player_x + 1, board_height - 5)], 
                         fill=(0, 0, 0, 100), width=1)
                # Ana çizgi
                draw.line([(player_x, 5), (player_x, board_height - 5)], 
                         fill=(120, 120, 150, 200), width=2)
            
            # Oyuncu etiketi
            try:
                if isinstance(player_id, int) and player_id < 0:
                    player_label = "---"
                else:
                    player_label = f"P{player_id}"
            except:
                player_label = f"P{player_id}"
            
            # Merkeze hizala
            bbox = draw.textbbox((0, 0), player_label, font=font_label)
            label_width = bbox[2] - bbox[0]
            label_x = player_x + (player_width - label_width) // 2
            
            # Gölge efekti (label) - daha belirgin
            draw.text((label_x + 2, 11), player_label, fill=(0, 0, 0, 200), font=font_label)
            draw.text((label_x, 9), player_label, fill=(220, 220, 240, 255), font=font_label)
            
            # Puan
            points = stats['points']
            points_text = f"{points:02d}"
            
            # Renk (sıralamaya göre)
            if idx == 0:
                score_color = (255, 215, 0, 255)  # Altın (RGB reversed for PIL)
            elif idx == 1:
                score_color = (192, 192, 192, 255)  # Gümüş
            else:
                score_color = (205, 162, 112, 255)  # Bronz
            
            # Puan merkezle
            bbox = draw.textbbox((0, 0), points_text, font=font_score)
            score_width = bbox[2] - bbox[0]
            score_x = player_x + (player_width - score_width) // 2
            
            # Gölge efekti (puan) - daha belirgin
            draw.text((score_x + 2, 30), points_text, fill=(0, 0, 0, 220), font=font_score)
            draw.text((score_x, 28), points_text, fill=score_color, font=font_score)
            
            # İstatistikler
            accuracy = (stats['makes'] / stats['attempts'] * 100) if stats['attempts'] > 0 else 0
            stats_text = f"{accuracy:.0f}%"
            
            bbox = draw.textbbox((0, 0), stats_text, font=font_stats)
            stats_width = bbox[2] - bbox[0]
            stats_x = player_x + (player_width - stats_width) // 2
            
            # Stats rengi (performansa göre)
            if accuracy >= 70:
                stats_color = (100, 255, 100, 255)  # Yeşil
            elif accuracy >= 50:
                stats_color = (255, 200, 100, 255)  # Turuncu
            else:
                stats_color = (255, 100, 100, 255)  # Kırmızı
            
            # Gölge efekti (stats)
            draw.text((stats_x + 1, 55), stats_text, fill=(0, 0, 0, 180), font=font_stats)
            draw.text((stats_x, 54), stats_text, fill=stats_color, font=font_stats)
        
        # PIL Image'ı numpy array'e çevir
        scoreboard_np = np.array(scoreboard_img)
        
        # RGBA'yı BGR'ye çevir (OpenCV formatı)
        scoreboard_bgr = cv2.cvtColor(scoreboard_np, cv2.COLOR_RGBA2BGRA)
        
        # Alpha channel'ı ayır
        alpha_channel = scoreboard_bgr[:, :, 3] / 255.0
        
        # Scoreboard pozisyonu
        start_x, start_y = 10, 10
        
        # Frame bölgesini al
        roi = self.frame[start_y:start_y + board_height, start_x:start_x + board_width]
        
        # Alpha blending
        for c in range(3):
            roi[:, :, c] = (alpha_channel * scoreboard_bgr[:, :, c] + 
                           (1 - alpha_channel) * roi[:, :, c])
        
        # Geri yerleştir
        self.frame[start_y:start_y + board_height, start_x:start_x + board_width] = roi

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
