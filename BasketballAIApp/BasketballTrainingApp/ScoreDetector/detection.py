
import cv2
import cvzone
import numpy as np
from matplotlib import pyplot as plt

hoop_detected = False  # Pota tespit edildi mi?
stable_hoop_pos = None  # Sabit pota pozisyonu (center, w, h)

def detection(frame,model_ball,playerModel):
    # --- HOOP DETECTION (sadece bir kez) ---
    if not hoop_detected:
        results_ball = model_ball(frame, stream=True, device=self.device)
        for r in results_ball:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                center = (x1 + w // 2, y1 + h // 2)
                current_class = ["basketball", "rim"][cls]

                if (conf > 0.3 or (
                        in_hoop_region(center, self.hoop_pos) and conf > 0.15)) and current_class == "basketball":
                    self.ball_pos.append((center, self.frame_count, w, h, conf))
                    cvzone.cornerRect(self.frame, (x1, y1, w, h))

                if conf > 0.5 and current_class == "rim":
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
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                center = (x1 + w // 2, y1 + h // 2)
                current_class = ["basketball", "rim"][cls]

                # Sadece basketbol topunu tespit et
                if (conf > 0.3 or (
                        in_hoop_region(center, self.hoop_pos) and conf > 0.15)) and current_class == "basketball":
                    self.ball_pos.append((center, self.frame_count, w, h, conf))
                    cvzone.cornerRect(self.frame, (x1, y1, w, h))

        # Sabit pota pozisyonunu hoop_pos listesinde kullan
        if self.stable_hoop_pos and (len(self.hoop_pos) == 0 or self.hoop_pos[-1][1] < self.frame_count):
            # Stable pozisyonu kullan
            self.hoop_pos = [self.stable_hoop_pos]

        # Potayı çiz (sabit pozisyon)
        hx, hy = self.stable_hoop_pos[0]
        hw, hh = self.stable_hoop_pos[2], self.stable_hoop_pos[3]
        cvzone.cornerRect(self.frame, (hx - hw // 2, hy - hh // 2, hw, hh), colorC=(0, 255, 0))

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
        cx, cy = int(l + w / 2), int(t + h)
        players.append((cx, cy, track_id))

        cv2.circle(self.frame, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(self.frame, f"P{track_id}", (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

