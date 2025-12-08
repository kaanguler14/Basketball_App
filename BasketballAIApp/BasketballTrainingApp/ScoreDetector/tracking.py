# shot_detector_deepsort.py
# OPTIMIZED VERSION:
# - FP16 Half Precision for YOLO
# - OpenCV Scoreboard (no PIL)
# - Async Video Reading (separate thread)

from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import time
from collections import deque
import logging
import threading
from queue import Queue
import gc
import torch

from utilsfixed import in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device
from deep_sort_realtime.deepsort_tracker import DeepSort
import pointSelection as ps
import homography as h
import draw_minimap as dm
from shot_detector import ShotDetectorModule

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

court_labels = [
    "top_left", "top_right","top", "bottom_right", "bottom_left",
    "free_throw_left", "free_throw_right",
    "center_circle", "paint_left", "paint_right"
]

vp_json = "video_points.json"
mp_json = "minimap_points.json"
scale = 0.5  # video resize


class AsyncVideoCapture:
    """
    Async Video Capture - Video okumayı ayrı thread'de yapar.
    Ana thread'i bloklamaz, FPS artışı sağlar.
    """
    def __init__(self, source, queue_size=2):
        self.cap = cv2.VideoCapture(source)
        self.queue = Queue(maxsize=queue_size)
        self.stopped = False
        
        # Thread başlat
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
    
    def _reader(self):
        """Arka planda frame okur"""
        while not self.stopped:
            if not self.queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.queue.put((ret, frame))
            else:
                time.sleep(0.001)  # Queue doluysa bekle
    
    def read(self):
        """Frame al"""
        if self.stopped and self.queue.empty():
            return False, None
        return self.queue.get()
    
    def release(self):
        """Kaynakları serbest bırak"""
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()


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
        
        # --- DEVICE ---
        self.device = get_device()
        self.use_half = self.device == "cuda"  # FP16 sadece CUDA'da
        
        # --- YOLO MODELLERİ ---
        self.model_ball = YOLO("D://repos//Basketball_App//BasketballAIApp//Trainings//kagglebest.pt")
        self.model_player = YOLO(r"D:\repos\Basketball_App\BasketballAIApp\Models\yolov8s.pt")
        
        logger.info(f"✓ Device: {self.device}, Half Precision: {self.use_half}")

        # --- DeepSORT Tracker ---
        # DeepSORT - OPTIMIZED: max_age düşürüldü (memory leak önleme)
        self.deepsort = DeepSort(
            max_age=15,  # 60 -> 15 (eski track'ler daha hızlı silinir)
            n_init=2,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3
        )
        self.deepsort_reset_interval = 500  # Her 500 frame'de tracker reset

        # --- ASYNC VIDEO CAPTURE ---
        video_path = r"D:\repos\Basketball_App\BasketballAIApp\clips\training7.mp4"
        self.cap = AsyncVideoCapture(video_path, queue_size=3)
        self.minimap_img = cv2.imread(r"D:\repos\Basketball_App\BasketballAIApp\BasketballTrainingApp\images\hom.png")
        self.frame_count = 0
        self.frame = None

        # --- BALL/HOOP LOGIC ---
        self.ball_pos = []
        self.hoop_pos = []
        
        # --- HOOP DETECTION OPTIMIZATION ---
        self.hoop_detected = False
        self.stable_hoop_pos = None

        # --- MODULAR SHOT DETECTOR ---
        self.shot_detector = ShotDetectorModule()

        # --- FPS ---
        self.prev_time = 0.0
        self.fps = 0
        self.fps_history = deque(maxlen=30)  # Smooth FPS

        # --- MINIMAP TOGGLE ---
        self.show_minimap = True
        
        # --- DETECTED PLAYERS ---
        self.detected_players = set()
        self.primary_player_id = None  # İlk tespit edilen oyuncu (tek oyuncu modunda sabit kalır)
        self.MAX_DETECTED_PLAYERS = 10  # Memory limit

        # --- SCOREBOARD COLORS (OpenCV BGR) ---
        self.colors = {
            'gold': (0, 215, 255),
            'silver': (192, 192, 192),
            'bronze': (112, 162, 205),
            'green': (100, 255, 100),
            'orange': (100, 200, 255),
            'red': (100, 100, 255),
            'white': (255, 255, 255),
            'dark_bg': (30, 25, 20),
        }

        # --- Homography ---
        # İlk frame için sync okuma (sadece bir kez)
        temp_cap = cv2.VideoCapture(video_path)
        ret, first_frame = temp_cap.read()
        temp_cap.release()
        
        first_frame = cv2.resize(first_frame, (int(first_frame.shape[1]*scale), int(first_frame.shape[0]*scale)))
        self.video_points_dict, self.minimap_points_dict = ps.load_or_select(first_frame)
        self.H, self.use_flip, self.h_img = h.compute_homography(self.video_points_dict, self.minimap_points_dict)

        self.run()

    # ------------------------ RUN ------------------------
    def run(self):
        cv2.namedWindow("Basketball Tracker", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Basketball Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break

            # --- FPS ---
            now = time.time()
            if self.prev_time:
                instant_fps = 1.0 / (now - self.prev_time)
                self.fps_history.append(instant_fps)
                self.fps = sum(self.fps_history) / len(self.fps_history)
            self.prev_time = now

            # Resize
            new_w = int(self.frame.shape[1] * scale)
            new_h = int(self.frame.shape[0] * scale)
            self.frame = cv2.resize(self.frame, (new_w, new_h))

            # --- DETECTION ---
            self._run_detection()

            # --- PLAYER DETECTION + DEEPSORT TRACKING ---
            players = self._run_player_tracking()

            # --- BALL & HOOP CLEANING + SHOT DETECTION ---
            self.clean_motion()
            self.shot_detection(players)

            # --- MINIMAP ---
            if self.show_minimap:
                self._draw_minimap_overlay(players)

            # --- SCORE OVERLAY ---
            if self.shot_detector.fade_counter > 0:
                cv2.putText(self.frame, self.shot_detector.overlay_text, (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, self.shot_detector.overlay_color, 4)
            
            # --- SCOREBOARD ---
            self._draw_scoreboard_opencv()

            # --- FPS Display ---
            cv2.putText(self.frame, f"FPS: {int(self.fps)}", (20, self.frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            minimap_status = "Minimap: ON (M)" if self.show_minimap else "Minimap: OFF (M)"
            cv2.putText(self.frame, minimap_status, (20, self.frame.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            self.frame_count += 1
            
            # MEMORY CLEANUP: Her 100 frame'de bir temizlik yap
            if self.frame_count % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # DEEPSORT RESET: Her 500 frame'de tracker'ı sıfırla (memory leak önleme)
            if self.frame_count % self.deepsort_reset_interval == 0 and self.frame_count > 0:
                self.deepsort = DeepSort(
                    max_age=15,
                    n_init=2,
                    nms_max_overlap=1.0,
                    max_cosine_distance=0.3
                )
                logger.info(f"🔄 DeepSORT reset (Frame {self.frame_count})")
            
            cv2.imshow("Basketball Tracker", self.frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('m') or key == ord('M'):
                self.show_minimap = not self.show_minimap

        self.cap.release()
        cv2.destroyAllWindows()

    def _run_detection(self):
        """Ball/Hoop detection with FP16 support"""
        # FP16 inference parameters
        inference_params = {
            'stream': True,
            'device': self.device,
            'half': self.use_half,
            'verbose': False
        }
        
        if not self.hoop_detected:
            results_ball = self.model_ball(self.frame, **inference_params)
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
                        
                        if not self.hoop_detected and len(self.hoop_pos) >= 5:
                            avg_cx = int(np.mean([pos[0][0] for pos in self.hoop_pos[-5:]]))
                            avg_cy = int(np.mean([pos[0][1] for pos in self.hoop_pos[-5:]]))
                            avg_w = int(np.mean([pos[2] for pos in self.hoop_pos[-5:]]))
                            avg_h = int(np.mean([pos[3] for pos in self.hoop_pos[-5:]]))
                            
                            self.stable_hoop_pos = ((avg_cx, avg_cy), self.frame_count, avg_w, avg_h, 1.0)
                            self.hoop_detected = True
                            logger.info(f"✓ Pota tespit edildi: {avg_cx}, {avg_cy}")
        else:
            results_ball = self.model_ball(self.frame, **inference_params)
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
            
            if self.stable_hoop_pos and (len(self.hoop_pos) == 0 or self.hoop_pos[-1][1] < self.frame_count):
                self.hoop_pos = [self.stable_hoop_pos]
                
            hx, hy = self.stable_hoop_pos[0]
            hw, hh = self.stable_hoop_pos[2], self.stable_hoop_pos[3]
            cvzone.cornerRect(self.frame, (hx - hw//2, hy - hh//2, hw, hh), colorC=(0, 255, 0))

    def _run_player_tracking(self):
        """Player detection + DeepSORT with FP16"""
        results_player = self.model_player(
            self.frame, 
            device=self.device, 
            conf=0.6,
            half=self.use_half,
            verbose=False
        )[0]
        
        CONF_THRESHOLD = 0.65
        detections = []
        
        if results_player.boxes is not None:
            # Batch processing for speed
            boxes_xyxy = results_player.boxes.xyxy.cpu().numpy()
            boxes_conf = results_player.boxes.conf.cpu().numpy()
            boxes_cls = results_player.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes_xyxy)):
                cls_name = results_player.names[boxes_cls[i]]
                if cls_name != "person":
                    continue
                    
                x1, y1, x2, y2 = map(int, boxes_xyxy[i])
                conf = float(boxes_conf[i])
                
                if conf >= CONF_THRESHOLD:
                    detections.append(((x1, y1, x2 - x1, y2 - y1), conf, boxes_cls[i]))
        
        tracks = self.deepsort.update_tracks(detections, frame=self.frame)
        
        players = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, w, h = track.to_ltwh()
            cx, cy = int(l + w/2), int(t + h)
            
            # TEK OYUNCU MODU: İlk tespit edilen ID'yi sabitle
            if self.num_players == 1:
                if self.primary_player_id is None:
                    self.primary_player_id = track_id
                    logger.info(f"✓ Primary player ID: P{track_id}")
                track_id = self.primary_player_id
            
            players.append((cx, cy, track_id))
            
            # Memory limit: Max 10 oyuncu ID'si tut
            if len(self.detected_players) < self.MAX_DETECTED_PLAYERS:
                self.detected_players.add(track_id)

            cv2.circle(self.frame, (cx, cy), 5, (0, 255, 255), -1)
            cv2.putText(self.frame, f"P{track_id}", (int(l), int(t) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        return players

    def _draw_minimap_overlay(self, players):
        """Minimap overlay"""
        minimap_display = dm.draw_minimap(
            self.minimap_img, self.shot_detector.shot_history,
            players, self.H, self.use_flip, self.h_img
        )
        
        minimap_scale = 0.2
        minimap_width = int(self.frame.shape[1] * minimap_scale)
        minimap_height = int(minimap_width * (minimap_display.shape[0] / minimap_display.shape[1]))
        minimap_small = cv2.resize(minimap_display, (minimap_width, minimap_height))
        
        margin = 20
        y_offset = self.frame.shape[0] - minimap_height - margin
        x_offset = self.frame.shape[1] - minimap_width - margin
        
        # Fast alpha blending using numpy
        alpha = 0.8
        roi = self.frame[y_offset:y_offset+minimap_height, x_offset:x_offset+minimap_width]
        cv2.addWeighted(minimap_small, alpha, roi, 1-alpha, 0, roi)
        
        cv2.rectangle(self.frame, (x_offset-2, y_offset-2), 
                     (x_offset+minimap_width+2, y_offset+minimap_height+2), 
                     (255, 255, 255), 2)

    def _draw_scoreboard_opencv(self):
        """
        OpenCV ile scoreboard çiz - PIL'den 10x daha hızlı!
        """
        player_scores = self.shot_detector.get_player_scores()
        
        # TEK OYUNCU MODU: Tüm skorları birleştir
        if self.num_players == 1 and len(player_scores) > 1:
            # Tüm skorları primary player'a birleştir
            total_points = sum(s['points'] for s in player_scores.values())
            total_makes = sum(s['makes'] for s in player_scores.values())
            total_attempts = sum(s['attempts'] for s in player_scores.values())
            
            display_id = self.primary_player_id if self.primary_player_id else list(player_scores.keys())[0]
            player_scores = {
                display_id: {
                    'points': total_points,
                    'makes': total_makes,
                    'attempts': total_attempts
                }
            }
        
        
        # Boyutlar
        if self.num_players == 1:
            board_width, board_height = 140, 65
        else:
            board_width, board_height = 240, 65
        
        start_x, start_y = 10, 10
        
        # Yarı saydam arka plan - SADECE ROI kopyala (10x daha hızlı)
        roi = self.frame[start_y:start_y+board_height, start_x:start_x+board_width].copy()
        cv2.rectangle(roi, (0, 0), (board_width, board_height), self.colors['dark_bg'], -1)
        cv2.addWeighted(roi, 0.85, 
                       self.frame[start_y:start_y+board_height, start_x:start_x+board_width], 
                       0.15, 0, 
                       self.frame[start_y:start_y+board_height, start_x:start_x+board_width])
        
        # Çerçeve
        cv2.rectangle(self.frame, (start_x, start_y), 
                     (start_x + board_width, start_y + board_height),
                     self.colors['white'], 2)
        
        # Player data - player_scores'daki TÜM oyuncuları dahil et
        all_players = {}
        
        # Önce player_scores'daki oyuncuları ekle (skor atanmış olanlar)
        for player_id, stats in player_scores.items():
            all_players[player_id] = stats
        
        # Sonra detected_players'dan eksik olanları ekle
        detected_list = sorted(list(self.detected_players))[:self.num_players]
        for player_id in detected_list:
            if player_id not in all_players:
                all_players[player_id] = {"points": 0, "makes": 0, "attempts": 0}
        
        # Hala oyuncu yoksa placeholder ekle
        if len(all_players) < self.num_players:
            for i in range(len(all_players), self.num_players):
                placeholder_id = -(i + 1)
                all_players[placeholder_id] = {"points": 0, "makes": 0, "attempts": 0}
        
        # En yüksek puanlı oyuncuları göster
        sorted_players = sorted(all_players.items(), key=lambda x: x[1]['points'], reverse=True)[:self.num_players]
        
        player_width = board_width // self.num_players
        
        for idx, (player_id, stats) in enumerate(sorted_players):
            player_x = start_x + idx * player_width
            center_x = player_x + player_width // 2
            
            # Dikey ayırıcı
            if idx > 0:
                cv2.line(self.frame, (player_x, start_y + 5), 
                        (player_x, start_y + board_height - 5), (150, 150, 150), 1)
            
            # Oyuncu etiketi
            label = "---" if isinstance(player_id, int) and player_id < 0 else f"P{player_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            label_x = center_x - label_size[0] // 2
            cv2.putText(self.frame, label, (label_x, start_y + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['white'], 1)
            
            # Puan
            points_text = f"{stats['points']:02d}"
            score_colors = [self.colors['gold'], self.colors['silver'], self.colors['bronze']]
            score_color = score_colors[min(idx, 2)]
            
            score_size = cv2.getTextSize(points_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            score_x = center_x - score_size[0] // 2
            cv2.putText(self.frame, points_text, (score_x, start_y + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, score_color, 2)
            
            # Yüzde
            accuracy = (stats['makes'] / stats['attempts'] * 100) if stats['attempts'] > 0 else 0
            pct_text = f"{accuracy:.0f}%"
            
            if accuracy >= 70:
                pct_color = self.colors['green']
            elif accuracy >= 50:
                pct_color = self.colors['orange']
            else:
                pct_color = self.colors['red']
            
            pct_size = cv2.getTextSize(pct_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            pct_x = center_x - pct_size[0] // 2
            cv2.putText(self.frame, pct_text, (pct_x, start_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, pct_color, 1)

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
            
        release_info = self.shot_detector.detect_shot(
            self.ball_pos, self.hoop_pos, players, 
            self.frame, self.frame_count
        )
        
        shot_data = self.shot_detector.score_shot(
            self.ball_pos, self.hoop_pos,
            self.H, self.use_flip, self.h_img,
            players
        )
        
        self.shot_detector.update_fade()
        
        return release_info, shot_data


if __name__ == "__main__":
    ShotDetector()
