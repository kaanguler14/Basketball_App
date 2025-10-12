"""
Shot Detection Module
=====================
Modular class for basketball shot detection and release point tracking.

This module was refactored from tracking.py to provide cleaner, more maintainable code
for shot detection logic. It handles:
- Dynamic threshold calculation based on camera perspective and shot distance
- Release point detection (when player releases the ball)
- Shot scoring (made/missed detection)
- Player ID tracking (which player took the shot)

Author: Basketball AI Team
Date: 2024
"""

import cv2
import math
import numpy as np
from utilsfixed import score, detect_down, detect_up


class ShotDetectorModule:
    """
    Şut tespiti ve release point detection için özelleşmiş class.
    
    This class encapsulates all shot detection logic including:
    - Dynamic threshold adjustment based on perspective and distance
    - Release point detection (moment when player releases the ball)
    - Shot scoring (made/missed shot detection)
    - Player ID tracking (which player took the shot)
    - Shot history management for minimap visualization
    
    The detector uses a multi-stage approach:
    1. Track ball-player distance over time
    2. Detect when ball is in player's hands (with dynamic thresholds)
    3. Detect release moment (when ball leaves hands)
    4. Track ball trajectory to hoop
    5. Score the shot (made/missed)
    
    Key Features:
    - Perspective compensation: Handles camera angle distortion
    - Distance-adaptive thresholds: Different thresholds for 3-point vs paint shots
    - Temporal smoothing: Uses frame history to reduce false positives
    - Robust release detection: Multiple criteria for reliable detection
    
    Attributes:
        ball_with_player (bool): Whether ball is currently in player's hands
        release_detected (bool): Whether a release has been detected
        shooter_id (int): ID of the player who took the shot
        makes (int): Number of successful shots
        attempts (int): Total number of shot attempts
        shot_history (list): History of shots for minimap visualization
    """
    
    def __init__(self):
        """Shot detector'ı başlat"""
        # Shot state
        self.ball_with_player = False
        self.release_detected = False
        self.release_frame = None
        self.release_player_pos = None
        self.shooter_id = None
        self.ball_player_history = []
        
        # Shot scoring state
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0
        self.makes = 0
        self.attempts = 0
        
        # Overlay
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_text = "waiting.."
        self.overlay_color = (0, 0, 0)
        
        # Shot history
        self.shot_history = []
        
        # Configuration
        self.HISTORY_SIZE = 15  # Frame history boyutu
        self.BASE_THRESHOLD = 50
    
    def detect_shot(self, ball_pos, hoop_pos, players, frame, frame_count):
        """
        Ana shot detection metodu.
        
        Args:
            ball_pos: Top pozisyonları listesi [(center, frame, w, h, conf), ...]
            hoop_pos: Pota pozisyonları listesi
            players: Oyuncu listesi [(cx, cy, id, w, h), ...]
            frame: Mevcut frame (opencv image)
            frame_count: Frame numarası
            
        Returns:
            dict: Shot detection sonuçları
        """
        if len(ball_pos) == 0:
            return None
        
        # Release point detection
        release_info = self._detect_release_point(
            ball_pos, hoop_pos, players, frame, frame_count
        )
        
        return release_info
    
    def _detect_release_point(self, ball_pos, hoop_pos, players, frame, frame_count):
        """
        Release point detection - topun oyuncunun elinden çıktığı anı tespit et.
        
        Returns:
            dict veya None: Release bilgileri (varsa)
        """
        if not players or len(ball_pos) == 0:
            return None
        
        bx, by = ball_pos[-1][0]
        
        # En yakın oyuncuyu bul
        nearest = min(players, key=lambda p: math.sqrt((p[0] - bx) ** 2 + (p[1] - by) ** 2))
        nearest_dist = math.sqrt((nearest[0] - bx) ** 2 + (nearest[1] - by) ** 2)
        px, py = nearest[0], nearest[1]
        
        # Top-oyuncu mesafe geçmişini kaydet
        self.ball_player_history.append({
            'frame': frame_count,
            'distance': nearest_dist,
            'player_pos': (px, py),
            'player_id': nearest[2] if len(nearest) > 2 else None,
            'ball_pos': (bx, by)
        })
        
        # History boyutunu sınırla
        if len(self.ball_player_history) > self.HISTORY_SIZE:
            self.ball_player_history.pop(0)
        
        # Potaya olan mesafeyi hesapla (dinamik threshold için)
        if len(hoop_pos) > 0:
            hx, hy = hoop_pos[-1][0]
            player_to_hoop_dist = math.sqrt((px - hx) ** 2 + (py - hy) ** 2)
        else:
            player_to_hoop_dist = 200  # default
        
        # Perspektif kompansasyonu
        frame_height = frame.shape[0]
        y_ratio = py / frame_height
        perspective_factor, depth_zone = self._calculate_perspective_factor(y_ratio)
        
        # Dinamik threshold hesapla
        HOLDING_THRESHOLD, RELEASE_THRESHOLD = self._calculate_dynamic_thresholds(
            player_to_hoop_dist, perspective_factor, y_ratio
        )
        
        # DEBUG LOG - Her frame'de göster
        print(f"[DEBUG] Mesafe: {nearest_dist:.1f}px, Threshold: {HOLDING_THRESHOLD}px, "
              f"Y-ratio: {y_ratio:.2f}, Factor: {perspective_factor:.2f}, "
              f"Hoop dist: {player_to_hoop_dist:.0f}px")
        
        # Top oyuncuya yakınsa
        if nearest_dist < HOLDING_THRESHOLD:
            if not self.ball_with_player:
                self.shooter_id = nearest[2] if len(nearest) > 2 else None
                print(f"   ✅ Top yakalandı: Oyuncu P{self.shooter_id} (mesafe: {nearest_dist:.1f}px < {HOLDING_THRESHOLD}px)")
            
            self.ball_with_player = True
            self.release_detected = False
            return None
        else:
            # DEBUG: Neden yakalanmadı?
            if not self.ball_with_player:
                print(f"   ❌ Top yakalanmadı: {nearest_dist:.1f}px >= {HOLDING_THRESHOLD}px (fark: {nearest_dist - HOLDING_THRESHOLD:.1f}px)")
        
        # Top oyuncudan uzaklaşıyorsa - RELEASE!
        if self.ball_with_player and not self.release_detected:
            release_info = self._check_release_condition(
                frame, frame_count, player_to_hoop_dist, 
                HOLDING_THRESHOLD, RELEASE_THRESHOLD,
                y_ratio, perspective_factor, depth_zone
            )
            return release_info
        
        return None
    
    def _calculate_perspective_factor(self, y_ratio):
        """
        Perspektif faktörünü hesapla - KAMERAYA YAKINLIK İÇİN İYİLEŞTİRİLMİŞ.
        
        Args:
            y_ratio: Y koordinatının frame yüksekliğine oranı
            
        Returns:
            tuple: (perspective_factor, depth_zone)
        """
        # DAHA YUMUŞAK perspektif faktörleri - aşırı büyütmeyi önle
        if y_ratio > 0.7:  # Kameraya çok yakın
            return 1.25, "ÖN"  # 1.5 -> 1.25 (daha yumuşak)
        elif y_ratio > 0.5:  # Orta-ön
            return 1.1, "ORTA-ÖN"  # 1.2 -> 1.1
        elif y_ratio > 0.3:  # Orta-arka
            return 1.0, "ORTA"
        else:  # Çok arkada
            return 0.85, "ARKA"  # 0.8 -> 0.85
    
    def _calculate_dynamic_thresholds(self, player_to_hoop_dist, perspective_factor, y_ratio):
        """
        Calculate dynamic thresholds for ball possession detection based on shot type and camera perspective.
        
        This method implements a multi-stage threshold calculation to handle perspective distortion:
        1. Base threshold selection based on player-to-hoop distance (shot type)
        2. Perspective compensation using squared factor (inverse square law)
        3. Extra boost for extreme camera proximity (y_ratio > 0.95)
        
        Why squared perspective factor?
        - Perspective distortion follows inverse square law: distortion ∝ 1/depth²
        - Objects closer to camera appear disproportionately larger
        - Distances between objects scale non-linearly with depth
        
        Why extra boost for high y_ratio?
        - When y_ratio > 0.95, player is very close to camera (bottom of frame)
        - Ball-hand distance can appear 150px+ even when physically in hand (~20cm)
        - Standard perspective factor (1.25²) only gives ~109px, insufficient for 150px
        - 2x boost brings threshold to 218px, covering extreme cases
        
        Args:
            player_to_hoop_dist (float): Distance from player to hoop in pixels
                - > 300px: 3-point shot (far)
                - 200-300px: Mid-range shot
                - 100-200px: Close-mid shot
                - < 100px: Paint area shot (very close)
            
            perspective_factor (float): Perspective compensation multiplier (0.85-1.25)
                - < 1.0: Player far from camera (top of frame)
                - > 1.0: Player close to camera (bottom of frame)
                - Calculated from y_ratio in _calculate_perspective_factor()
            
            y_ratio (float): Vertical position in frame (0.0 = top, 1.0 = bottom)
                - Used for extreme proximity detection
                - > 0.95: Very close to camera, needs 2x boost
                - > 0.85: Moderately close, needs 1.5x boost
        
        Returns:
            tuple: (HOLDING_THRESHOLD, RELEASE_THRESHOLD)
                - HOLDING_THRESHOLD (int): Max distance (px) to consider ball in player's hand
                - RELEASE_THRESHOLD (int): Min distance increase (px) to detect ball release
        
        Example:
            >>> # Player close to camera, 3-point shot
            >>> holding, release = _calculate_dynamic_thresholds(398, 1.25, 0.99)
            >>> # holding = 70 * 1.25² * 2.0 = 218px (can detect 150px actual distance)
            >>> # release = 15 / 1.25 = 12px (easier release detection when close)
        """
        # STEP 1: Select base threshold based on shot type (player-to-hoop distance)
        # Rationale: Players farther from hoop appear smaller, so ball-hand distance appears larger
        if player_to_hoop_dist > 300:  # 3-point shot (far from hoop)
            holding = 70  # Higher threshold: player appears small, ball-hand distance ~50-70px
            release = 15  # Lower threshold: small movements are significant        elif player_to_hoop_dist > 200:  # Mid-range shot
            holding = 55  # Medium threshold: medium-sized player
            release = 20  # Medium sensitivity
        elif player_to_hoop_dist > 100:  # Close-mid shot
            holding = 45  # Lower threshold: player appears larger
            release = 25  # Higher threshold: need more movement to confirm release
        else:  # Paint area shot (< 100px, very close to hoop)
            holding = 35  # Lowest threshold: player appears very large, ball-hand distance ~20-35px
            release = 30  # Highest threshold: close shots are slower, need significant movement
        
        # STEP 2: Apply perspective compensation (SQUARED factor for inverse square law)
        # Why squared? Perspective distortion ∝ 1/depth²
        # - Ball closer to camera → appears disproportionately farther from hand
        # - Example: 1.25² = 1.5625x amplification (70px → 109px)
        holding = int(holding * perspective_factor * perspective_factor)
        
        # STEP 3: Extra boost for extreme camera proximity
        # Problem: Even with squared factor, very close players (y_ratio > 0.95) have
        # ball-hand distances of 150px+, but squared factor only gives ~109px
        # Solution: Additional multiplier for extreme cases
        if y_ratio > 0.95:  # Very close to camera (bottom 5% of frame)
            holding = int(holding * 2.0)  # 2x boost → 109px * 2.0 = 218px (covers 150px actual distance)
        elif y_ratio > 0.85:  # Moderately close to camera (bottom 15% of frame)
            holding = int(holding * 1.5)  # 1.5x boost → handles 120-140px distances
        
        # STEP 4: Adjust release threshold (inverse relationship)
        # Why divide? Ball closer to camera moves faster in pixel space
        # → Smaller pixel movement = actual release → lower threshold needed
        release = int(release / perspective_factor)
        
        return holding, release
    
    def _check_release_condition(self, frame, frame_count, player_to_hoop_dist,
                                 HOLDING_THRESHOLD, RELEASE_THRESHOLD,
                                 y_ratio, perspective_factor, depth_zone):
        """
        Release condition kontrolü - topun gerçekten elden çıktığını doğrula.
        
        Returns:
            dict veya None: Release bilgileri
        """
        if len(self.ball_player_history) < 2:
            return None
        
        # Son frame'leri al
        recent = self.ball_player_history[-2:]
        
        # Mesafe artışı
        dist_increase = recent[-1]['distance'] - recent[0]['distance']
        
        # Y hareketi (yukarı = negatif)
        y_movement = recent[-1]['ball_pos'][1] - recent[0]['ball_pos'][1]
        
        # Hız hesapla
        ball_velocity = math.sqrt(
            (recent[-1]['ball_pos'][0] - recent[0]['ball_pos'][0]) ** 2 +
            (recent[-1]['ball_pos'][1] - recent[0]['ball_pos'][1]) ** 2
        )
        
        # Release kriterleri - MESAFE VE KAMERAYA YAKINLIK İÇİN ADAPTIF
        # Kameraya çok yakın pozisyonlarda (y_ratio > 0.7) daha esnek ol
        is_close_to_camera = y_ratio > 0.7
        
        if is_close_to_camera:
            # KAMERAYA YAKIN - Perspektif etkisi nedeniyle çok esnek kriterler
            release_condition = (
                dist_increase > RELEASE_THRESHOLD * 0.8 or  # %20 daha esnek
                (y_movement < -3 and ball_velocity > 6) or   # Çok düşük gereksinimler
                ball_velocity > 10  # Minimum hız çok düşük
            )
        elif player_to_hoop_dist < 100:  # ÇOK YAKIN ŞUT (ama kameraya uzak)
            release_condition = (
                dist_increase > RELEASE_THRESHOLD or
                (y_movement < -5 and ball_velocity > 8) or
                ball_velocity > 12
            )
        elif player_to_hoop_dist < 200:  # YAKIN-ORTA
            release_condition = (
                dist_increase > RELEASE_THRESHOLD or
                (y_movement < -8 and ball_velocity > 12) or
                ball_velocity > 18
            )
        else:  # UZAK ŞUT
            release_condition = (
                dist_increase > RELEASE_THRESHOLD or
                (y_movement < -10 and ball_velocity > 15) or
                ball_velocity > 25
            )
        
        if not release_condition:
            return None
        
        # RELEASE TESPİT EDİLDİ!
        self.release_detected = True
        self.release_frame = frame_count
        
        # Shooter pozisyonunu bul
        release_pos, shooter_frames = self._find_shooter_position(HOLDING_THRESHOLD)
        self.release_player_pos = release_pos
        
        # Shot type belirle - DAHA DETAYLI
        if player_to_hoop_dist > 300:
            shot_type = "UZAK (3PT)"
        elif player_to_hoop_dist > 200:
            shot_type = "ORTA"
        elif player_to_hoop_dist > 100:
            shot_type = "YAKIN"
        else:
            shot_type = "PAINT"  # Çok yakın şutlar için özel kategori
        
        # Log
        self._log_release_info(
            frame_count, shot_type, depth_zone, y_ratio,
            dist_increase, ball_velocity, player_to_hoop_dist,
            HOLDING_THRESHOLD, RELEASE_THRESHOLD, perspective_factor,
            shooter_frames
        )
        
        # Frame'de görsel işaretleme
        self._draw_release_marker(
            frame, shot_type, depth_zone, 
            HOLDING_THRESHOLD, RELEASE_THRESHOLD,
            shooter_frames
        )
        
        return {
            'frame': frame_count,
            'position': self.release_player_pos,
            'shooter_id': self.shooter_id,
            'shot_type': shot_type,
            'verified': len(shooter_frames) > 0
        }
    
    def _find_shooter_position(self, HOLDING_THRESHOLD):
        """
        Shooter'ın pozisyonunu history'den bul.
        
        Returns:
            tuple: (position, shooter_frames)
        """
        best_idx = -1
        shooter_frames = []
        
        # Geriye doğru git ve shooter_id'ye sahip frame'leri bul
        for i in range(len(self.ball_player_history) - 1, -1, -1):
            frame_data = self.ball_player_history[i]
            
            if frame_data['player_id'] == self.shooter_id:
                if frame_data['distance'] < HOLDING_THRESHOLD * 1.3:
                    shooter_frames.append(i)
                    if best_idx == -1:
                        best_idx = i
            
            # Son 8 frame yeterli
            if len(shooter_frames) >= 5:
                break
        
        # Fallback: Eğer shooter bulunamazsa
        if best_idx == -1:
            for i in range(len(self.ball_player_history) - 1, 
                          max(0, len(self.ball_player_history) - 4), -1):
                if self.ball_player_history[i]['distance'] < HOLDING_THRESHOLD * 1.2:
                    best_idx = i
                    break
        
        # Pozisyonu döndür
        if best_idx != -1:
            position = self.ball_player_history[best_idx]['player_pos']
            # Shooter ID'yi doğrula
            confirmed_id = self.ball_player_history[best_idx]['player_id']
            if confirmed_id is not None:
                self.shooter_id = confirmed_id
        else:
            # En son pozisyon
            position = self.ball_player_history[-1]['player_pos']
        
        return position, shooter_frames
    
    def _log_release_info(self, frame_count, shot_type, depth_zone, y_ratio,
                         dist_increase, ball_velocity, player_to_hoop_dist,
                         HOLDING_THRESHOLD, RELEASE_THRESHOLD, perspective_factor,
                         shooter_frames):
        """Release bilgilerini logla"""
        camera_proximity = "KAMERAYA YAKIN" if y_ratio > 0.7 else ""
        print(f"🏀 {shot_type} ŞUT ATILDI! [Derinlik: {depth_zone}] {camera_proximity}")
        print(f"   Frame: {frame_count}, Oyuncu: P{self.shooter_id} "
              f"({'✓ Doğrulandı' if len(shooter_frames) > 0 else '⚠ Fallback'})")
        print(f"   Pozisyon: {self.release_player_pos}, Y-oranı: {y_ratio:.2f}")
        print(f"   Mesafe artışı: {dist_increase:.1f}px, Hız: {ball_velocity:.1f}px/f")
        print(f"   Potaya mesafe: {player_to_hoop_dist:.0f}px")
        print(f"   Threshold: HOLDING={HOLDING_THRESHOLD}px, "
              f"RELEASE={RELEASE_THRESHOLD}px (faktör={perspective_factor:.2f})")
        print(f"   Shooter frame sayısı: {len(shooter_frames)} (aynı oyuncuya ait)")
    
    def _draw_release_marker(self, frame, shot_type, depth_zone,
                            HOLDING_THRESHOLD, RELEASE_THRESHOLD, shooter_frames):
        """Frame üzerine release marker'ı çiz"""
        if not self.release_player_pos:
            return
        
        x, y = self.release_player_pos
        
        # Ana işaret
        cv2.circle(frame, (x, y), 15, (255, 0, 255), 3)
        
        # Oyuncu ID ve shot type
        cv2.putText(frame, f"P{self.shooter_id} RELEASE ({shot_type})", 
                   (x - 80, y - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Zone ve threshold bilgisi
        cv2.putText(frame, f"{depth_zone} | H:{HOLDING_THRESHOLD} R:{RELEASE_THRESHOLD}", 
                   (x - 80, y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Doğrulama işareti
        if len(shooter_frames) > 0:
            cv2.putText(frame, "VERIFIED", 
                       (x - 30, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    def score_shot(self, ball_pos, hoop_pos, H, use_flip, h_img, players):
        """
        Şutu skorla (başarılı/başarısız).
        
        Returns:
            dict veya None: Scoring bilgileri
        """
        if len(hoop_pos) == 0 or len(ball_pos) == 0:
            return None
        
        # Up/Down detection
        if not self.up:
            self.up = detect_up(ball_pos, hoop_pos)
            if self.up:
                self.up_frame = ball_pos[-1][1]
        
        if self.up and not self.down:
            self.down = detect_down(ball_pos, hoop_pos)
            if self.down:
                self.down_frame = ball_pos[-1][1]
        
        # Şut tamamlandı mı?
        if self.up and self.down and self.up_frame < self.down_frame:
            self.attempts += 1
            made = score(ball_pos, hoop_pos)
            
            if made:
                self.makes += 1
                self.overlay_text = "Score"
                self.overlay_color = (0, 255, 0)
            else:
                self.overlay_text = "Miss"
                self.overlay_color = (0, 0, 255)
            
            self.fade_counter = self.fade_frames
            
            # Minimap için shot pozisyonunu kaydet
            shot_data = self._record_shot_for_minimap(
                made, H, use_flip, h_img, players
            )
            
            # Reset
            self.up = False
            self.down = False
            self.ball_with_player = False
            self.release_detected = False
            self.release_player_pos = None
            self.shooter_id = None
            self.ball_player_history = []
            
            return shot_data
        
        return None
    
    def _record_shot_for_minimap(self, made, H, use_flip, h_img, players):
        """Şutu minimap için kaydet"""
        if self.release_player_pos and self.shooter_id is not None:
            # Release pozisyonunu kullan
            pt = np.array([[[self.release_player_pos[0], 
                           self.release_player_pos[1]]]], dtype=np.float32)
            proj_pt = cv2.perspectiveTransform(pt, H)[0][0]
            mx = int(proj_pt[0])
            my = int(proj_pt[1])
            
            if use_flip:
                my = h_img - my
            
            self.shot_history.append((mx, my, made, self.shooter_id))
            print(f"📍 Minimap'e eklendi: ({mx}, {my}), "
                  f"Oyuncu: P{self.shooter_id}, "
                  f"{'BAŞARILI' if made else 'BAŞARISIZ'}")
            
            return {
                'minimap_pos': (mx, my),
                'made': made,
                'shooter_id': self.shooter_id
            }
        else:
            # Fallback: Mevcut oyuncu pozisyonunu kullan
            if self.shooter_id is not None and players:
                for p in players:
                    if len(p) > 2 and p[2] == self.shooter_id:
                        pt = np.array([[[p[0], p[1]]]], dtype=np.float32)
                        proj_pt = cv2.perspectiveTransform(pt, H)[0][0]
                        mx = int(proj_pt[0])
                        my = int(proj_pt[1])
                        
                        if use_flip:
                            my = h_img - my
                        
                        self.shot_history.append((mx, my, made, self.shooter_id))
                        print(f"📍 Minimap'e eklendi (FALLBACK): ({mx}, {my}), "
                              f"Oyuncu: P{self.shooter_id}")
                        
                        return {
                            'minimap_pos': (mx, my),
                            'made': made,
                            'shooter_id': self.shooter_id
                        }
        
        return None
    
    def update_fade(self):
        """Fade counter'ı güncelle"""
        if self.fade_counter > 0:
            self.fade_counter -= 1
    
    def get_stats(self):
        """İstatistikleri döndür"""
        return {
            'makes': self.makes,
            'attempts': self.attempts,
            'percentage': (self.makes / self.attempts * 100) if self.attempts > 0 else 0
        }

