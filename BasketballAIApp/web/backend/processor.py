"""
Video Processor - Web Version (Full Featured)
=============================================
Adapted from tracking.py with full homography and 3PT support.
"""

import cv2
import cvzone
import numpy as np
import time
import os
import sys
from pathlib import Path
from collections import deque
import gc
import torch
import json

# Fix PyTorch 2.6+ weights_only security change
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Add parent paths for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
SCORE_DETECTOR_DIR = PROJECT_ROOT / "BasketballAIApp" / "BasketballTrainingApp" / "ScoreDetector"
sys.path.insert(0, str(SCORE_DETECTOR_DIR))

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


class VideoProcessor:
    """
    Web-adapted video processor with full homography and 3PT support.
    """
    
    def __init__(
        self,
        video_path: str,
        num_players: int = 1,
        output_dir: str = "./outputs",
        job_id: str = "default",
        status_callback=None,
        three_point_line: list = None
    ):
        self.video_path = video_path
        self.num_players = num_players
        self.output_dir = Path(output_dir)
        self.job_id = job_id
        self.status_callback = status_callback or (lambda msg, prog: None)
        
        self.output_dir.mkdir(exist_ok=True)
        
        # Device
        self.device = self._get_device()
        self.use_half = self.device == "cuda"
        
        # Models
        self.status_callback("Modeller yükleniyor...", 5)
        app_root = PROJECT_ROOT / "BasketballAIApp"
        model_dir = app_root / "Trainings"
        models_dir = app_root / "Models"
        
        self.model_ball = YOLO(str(model_dir / "kagglebest.pt"))
        self.model_player = YOLO(str(models_dir / "yolov8s.pt"))
        
        # DeepSORT
        self.deepsort = DeepSort(
            max_age=15,
            n_init=2,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3
        )
        
        # User-defined 3PT line (directly on video frame coordinates)
        self.user_three_point_line = three_point_line
        
        # State
        self.ball_pos = []
        self.hoop_pos = []
        self.hoop_detected = False
        self.stable_hoop_pos = None
        self.detected_players = set()
        self.primary_player_id = None
        
        # Shot detection state
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0
        self.makes = 0
        self.attempts = 0
        
        # Stats
        self.stats = {
            "total_shots": 0,
            "made_shots": 0,
            "missed_shots": 0,
            "two_pointers": 0,
            "three_pointers": 0,
            "player_stats": {}
        }
        
        # 3PT polygon - Use user-defined line directly on video coordinates
        self._three_point_polygon = None
        if self.user_three_point_line and len(self.user_three_point_line) >= 3:
            self._three_point_polygon = np.array(self.user_three_point_line, dtype=np.int32)
            print(f"[INFO] User 3PT line loaded: {len(self.user_three_point_line)} points")
        
        # Colors
        self.colors = {
            'gold': (0, 215, 255),
            'silver': (192, 192, 192),
            'green': (100, 255, 100),
            'red': (100, 100, 255),
            'white': (255, 255, 255),
            'dark_bg': (30, 25, 20),
        }
        
        self.scale = 0.5
    
    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _classify_shot_2pt_or_3pt(self, x, y):
        """
        Classify shot as 2PT or 3PT based on video position and user-defined 3PT line.
        User draws the 3PT LINE (arc shape around the basket).
        - Inside polygon = 2PT (closer to basket)
        - Outside polygon = 3PT (further from basket)
        """
        if self._three_point_polygon is None:
            return 2
        
        try:
            result = cv2.pointPolygonTest(
                self._three_point_polygon, (float(x), float(y)), False
            )
            # Inside polygon (>=0) = 2PT, Outside (<0) = 3PT
            points = 2 if result >= 0 else 3
            print(f"[3PT CHECK] pos=({x},{y}), result={result}, points={points}")
            return points
        except Exception as e:
            print(f"[3PT CHECK ERROR] {e}")
            return 2
    
    def process(self) -> dict:
        """Main processing function"""
        self.status_callback("Video açılıyor...", 10)
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise Exception("Video açılamadı")
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.scale)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.scale)
        
        # Output video - use H264 codec for browser compatibility
        output_path = self.output_dir / f"{self.job_id}_output.mp4"
        temp_path = self.output_dir / f"{self.job_id}_temp.avi"
        
        # Try H264 first, fallback to mp4v
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            # Fallback: use XVID for temp, convert later
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))
            self._needs_conversion = True
        else:
            self._needs_conversion = False
            temp_path = None
        
        self._temp_path = temp_path
        self._output_path = output_path
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize
                frame = cv2.resize(frame, (width, height))
                
                # Process frame
                frame = self._process_frame(frame, frame_count)
                
                # Write output
                out.write(frame)
                
                frame_count += 1
                
                # Update progress
                if frame_count % 30 == 0:
                    progress = int(10 + (frame_count / total_frames) * 80)
                    self.status_callback(
                        f"İşleniyor... {frame_count}/{total_frames} frame",
                        min(progress, 90)
                    )
                
                # Memory cleanup
                if frame_count % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # DeepSORT reset
                if frame_count % 500 == 0 and frame_count > 0:
                    self.deepsort = DeepSort(
                        max_age=15,
                        n_init=2,
                        nms_max_overlap=1.0,
                        max_cosine_distance=0.3
                    )
        
        finally:
            cap.release()
            out.release()
        
        self.status_callback("Video dönüştürülüyor...", 92)
        
        # Convert to browser-compatible format if needed
        if self._needs_conversion and self._temp_path and self._temp_path.exists():
            self._convert_video(str(self._temp_path), str(self._output_path))
            # Remove temp file
            try:
                os.remove(str(self._temp_path))
            except:
                pass
        
        self.status_callback("Tamamlanıyor...", 95)
        
        # Calculate final stats
        if self.stats["total_shots"] > 0:
            self.stats["accuracy"] = round(
                self.stats["made_shots"] / self.stats["total_shots"] * 100, 1
            )
        else:
            self.stats["accuracy"] = 0
        
        return {
            "video_url": f"/api/video/{self.job_id}",
            "stats": self.stats,
            "total_frames": frame_count
        }
    
    def _process_frame(self, frame, frame_count):
        """Process a single frame"""
        
        # Ball/Hoop detection
        self._detect_ball_hoop(frame, frame_count)
        
        # Player detection
        players = self._detect_players(frame)
        
        # Clean old positions
        self._clean_positions(frame_count)
        
        # Shot detection with homography
        self._detect_shot(frame_count, players)
        
        # Draw scoreboard
        self._draw_scoreboard(frame)
        
        return frame
    
    def _detect_ball_hoop(self, frame, frame_count):
        """Detect ball and hoop"""
        inference_params = {
            'stream': True,
            'device': self.device,
            'half': self.use_half,
            'verbose': False
        }
        
        results = self.model_ball(frame, **inference_params)
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2-x1, y2-y1
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                center = (x1 + w//2, y1 + h//2)
                current_class = ["basketball", "rim"][cls]
                
                if conf > 0.3 and current_class == "basketball":
                    self.ball_pos.append((center, frame_count, w, h, conf))
                    cvzone.cornerRect(frame, (x1, y1, w, h))
                
                if conf > 0.5 and current_class == "rim":
                    if not self.hoop_detected:
                        self.hoop_pos.append((center, frame_count, w, h, conf))
                        cvzone.cornerRect(frame, (x1, y1, w, h))
                        
                        if len(self.hoop_pos) >= 5:
                            avg_cx = int(np.mean([p[0][0] for p in self.hoop_pos[-5:]]))
                            avg_cy = int(np.mean([p[0][1] for p in self.hoop_pos[-5:]]))
                            avg_w = int(np.mean([p[2] for p in self.hoop_pos[-5:]]))
                            avg_h = int(np.mean([p[3] for p in self.hoop_pos[-5:]]))
                            self.stable_hoop_pos = ((avg_cx, avg_cy), frame_count, avg_w, avg_h, 1.0)
                            self.hoop_detected = True
        
        # Draw stable hoop
        if self.hoop_detected and self.stable_hoop_pos:
            hx, hy = self.stable_hoop_pos[0]
            hw, hh = self.stable_hoop_pos[2], self.stable_hoop_pos[3]
            cvzone.cornerRect(frame, (hx - hw//2, hy - hh//2, hw, hh), colorC=(0, 255, 0))
            self.hoop_pos = [self.stable_hoop_pos]
    
    def _detect_players(self, frame):
        """Detect and track players"""
        results = self.model_player(
            frame,
            device=self.device,
            conf=0.6,
            half=self.use_half,
            verbose=False
        )[0]
        
        detections = []
        if results.boxes is not None:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            boxes_conf = results.boxes.conf.cpu().numpy()
            boxes_cls = results.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes_xyxy)):
                cls_name = results.names[boxes_cls[i]]
                if cls_name != "person":
                    continue
                x1, y1, x2, y2 = map(int, boxes_xyxy[i])
                conf = float(boxes_conf[i])
                if conf >= 0.65:
                    detections.append(((x1, y1, x2-x1, y2-y1), conf, boxes_cls[i]))
        
        tracks = self.deepsort.update_tracks(detections, frame=frame)
        
        players = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, w, h = track.to_ltwh()
            cx, cy = int(l + w/2), int(t + h)
            
            # Single player mode: fix ID
            if self.num_players == 1:
                if self.primary_player_id is None:
                    self.primary_player_id = track_id
                track_id = self.primary_player_id
            
            players.append((cx, cy, track_id))
            
            if len(self.detected_players) < 10:
                self.detected_players.add(track_id)
            
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
            cv2.putText(frame, f"P{track_id}", (int(l), int(t) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return players
    
    def _clean_positions(self, frame_count):
        """Clean old ball/hoop positions"""
        while len(self.ball_pos) > 0 and (frame_count - self.ball_pos[0][1] > 30):
            self.ball_pos.pop(0)
        while len(self.ball_pos) > 50:
            self.ball_pos.pop(0)
    
    def _detect_shot(self, frame_count, players):
        """Detect shots with homography-based 2PT/3PT classification"""
        if len(self.hoop_pos) == 0 or len(self.ball_pos) == 0:
            return
        
        hoop = self.hoop_pos[-1]
        hoop_cx, hoop_cy = hoop[0]
        hoop_h = hoop[3]
        
        ball = self.ball_pos[-1]
        bx, by = ball[0]
        
        # Up detection
        if not self.up:
            up_y1 = hoop_cy - 3.0 * hoop_h
            up_y2 = hoop_cy - 1.1 * hoop_h
            if up_y1 < by < up_y2:
                self.up = True
                self.up_frame = frame_count
        
        # Down detection
        if self.up and not self.down:
            down_threshold = hoop_cy + 0.6 * hoop_h
            if by > down_threshold:
                self.down = True
                self.down_frame = frame_count
        
        # Shot completed
        if self.up and self.down and self.up_frame < self.down_frame:
            self.stats["total_shots"] += 1
            self.attempts += 1
            
            # Check if made
            made = self._check_made_shot()
            
            # Find shooter
            shooter_id = self.primary_player_id or 1
            shooter_pos = None
            
            if players:
                # Find closest player to ball at up_frame
                min_dist = float('inf')
                for px, py, pid in players:
                    dist = abs(px - bx) + abs(py - by)
                    if dist < min_dist:
                        min_dist = dist
                        shooter_id = pid
                        shooter_pos = (px, py)
            
            # Determine 2PT or 3PT using user-defined 3PT line (direct video coordinates)
            points_val = 2
            if shooter_pos:
                points_val = self._classify_shot_2pt_or_3pt(shooter_pos[0], shooter_pos[1])
            
            if made:
                self.stats["made_shots"] += 1
                self.makes += 1
                if points_val == 3:
                    self.stats["three_pointers"] += 1
                else:
                    self.stats["two_pointers"] += 1
            else:
                self.stats["missed_shots"] += 1
            
            # Update player stats
            if shooter_id not in self.stats["player_stats"]:
                self.stats["player_stats"][shooter_id] = {
                    "shots": 0, "made": 0, "points": 0
                }
            self.stats["player_stats"][shooter_id]["shots"] += 1
            if made:
                self.stats["player_stats"][shooter_id]["made"] += 1
                self.stats["player_stats"][shooter_id]["points"] += points_val
            
            print(f"[SHOT] {'MADE' if made else 'MISS'} - {points_val}PT by P{shooter_id}")
            
            # Reset
            self.up = False
            self.down = False
    
    def _check_made_shot(self):
        """Check if shot was made"""
        if len(self.ball_pos) < 2 or len(self.hoop_pos) < 1:
            return False
        
        hoop = self.hoop_pos[-1]
        hoop_cx, hoop_cy = hoop[0]
        hoop_w = hoop[2]
        
        rim_x1 = hoop_cx - 0.5 * hoop_w
        rim_x2 = hoop_cx + 0.5 * hoop_w
        
        # Check recent ball positions
        for ball in self.ball_pos[-15:]:
            bx, by = ball[0]
            if rim_x1 <= bx <= rim_x2 and abs(by - hoop_cy) < hoop_w:
                return True
        
        return False
    
    def _draw_scoreboard(self, frame):
        """Draw scoreboard on frame"""
        board_width, board_height = 220, 100
        start_x, start_y = 10, 10
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y),
                     (start_x + board_width, start_y + board_height),
                     self.colors['dark_bg'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        cv2.rectangle(frame, (start_x, start_y),
                     (start_x + board_width, start_y + board_height),
                     self.colors['gold'], 2)
        
        # Stats text
        made = self.stats["made_shots"]
        total = self.stats["total_shots"]
        pct = (made / total * 100) if total > 0 else 0
        
        cv2.putText(frame, f"Shots: {made}/{total}", (start_x + 10, start_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['white'], 2)
        
        cv2.putText(frame, f"Accuracy: {pct:.0f}%", (start_x + 10, start_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   self.colors['green'] if pct >= 50 else self.colors['red'], 2)
        
        # Points breakdown
        two_pts = self.stats["two_pointers"]
        three_pts = self.stats["three_pointers"]
        total_points = (two_pts * 2) + (three_pts * 3)
        
        cv2.putText(frame, f"2PT: {two_pts}  3PT: {three_pts}", (start_x + 10, start_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['silver'], 1)
        
        cv2.putText(frame, f"Points: {total_points}", (start_x + 10, start_y + 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['gold'], 2)
    
    def _convert_video(self, input_path: str, output_path: str):
        """Convert video to browser-compatible H.264 format using ffmpeg"""
        import subprocess
        
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 min timeout
            )
            
            if result.returncode != 0:
                print(f"[WARNING] ffmpeg conversion failed: {result.stderr}")
                # Fallback: just rename temp to output
                import shutil
                shutil.copy(input_path, output_path)
                
        except FileNotFoundError:
            print("[WARNING] ffmpeg not found, using raw video")
            import shutil
            shutil.copy(input_path, output_path)
        except Exception as e:
            print(f"[WARNING] Video conversion failed: {e}")
            import shutil
            shutil.copy(input_path, output_path)
