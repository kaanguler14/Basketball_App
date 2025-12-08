"""
Shot Detection Module
=====================
Shot tespiti ve release point detection için modüler class.
tracking.py'den ayrıştırıldı - daha temiz ve maintainable kod.

OPTIMIZED VERSION:
- JSON caching (3PT polygon bir kez yükleniyor)
- deque kullanımı (O(1) pop işlemi)
- Logging ile kontrollü debug output
"""

import cv2
import math
import numpy as np
import json
import os
import logging
from collections import deque
from utilsfixed import score, detect_down, detect_up

# Logging konfigürasyonu
logger = logging.getLogger(__name__)


class ShotDetectorModule:
    """
    Şut tespiti ve release point detection için özelleşmiş class.

    Özellikler:
    - Dinamik threshold ayarlama (perspektif ve mesafeye göre)
    - Release point detection (oyuncunun topu bıraktığı an)
    - Shot scoring (başarılı/başarısız şut tespiti)
    - Player ID tracking (hangi oyuncu şutu attı)
    """

    def __init__(self):
        """Shot detector'ı başlat"""
        # Shot state
        self.ball_with_player = False
        self.release_detected = False
        self.release_frame = None
        self.release_player_pos = None
        self.shooter_id = None
        
        # OPTIMIZED: deque kullanımı - O(1) pop işlemi
        self.HISTORY_SIZE = 15
        self.ball_player_history = deque(maxlen=self.HISTORY_SIZE)

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

        # Shot history - MAX 100 kayıt (memory leak önleme)
        self.shot_history = []
        self.MAX_SHOT_HISTORY = 100
        
        # Player scores tracking
        self.player_scores = {}

        # Configuration
        self.BASE_THRESHOLD = 50
        
        # OPTIMIZED: 3PT polygon cache - bir kez yükle
        self._three_point_polygon = None
        self._load_three_point_polygon()

    def _load_three_point_polygon(self):
        """3PT çizgisi poligonunu bir kez yükle ve cache'le"""
        try:
            base_dir = os.path.dirname(__file__)
            json_path = os.path.join(base_dir, "three_point_line.json")
            
            if not os.path.exists(json_path):
                logger.warning(f"3PT dosyası bulunamadı: {json_path}")
                self._three_point_polygon = None
                return
            
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            points_3pt = data.get("points", [])
            if len(points_3pt) >= 3:
                self._three_point_polygon = np.array(points_3pt, dtype=np.float32)
                logger.info(f"✓ 3PT polygon yüklendi: {len(points_3pt)} nokta")
            else:
                logger.warning(f"3PT çizgisi yetersiz nokta: {len(points_3pt)}")
                self._three_point_polygon = None
                
        except Exception as e:
            logger.error(f"3PT polygon yükleme hatası: {e}")
            self._three_point_polygon = None

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
        nearest = min(
            players, key=lambda p: math.sqrt((p[0] - bx) ** 2 + (p[1] - by) ** 2)
        )
        nearest_dist = math.sqrt((nearest[0] - bx) ** 2 + (nearest[1] - by) ** 2)
        px, py = nearest[0], nearest[1]

        # Top-oyuncu mesafe geçmişini kaydet (deque otomatik olarak eski verileri siler)
        self.ball_player_history.append(
            {
                "frame": frame_count,
                "distance": nearest_dist,
                "player_pos": (px, py),
                "player_id": nearest[2] if len(nearest) > 2 else None,
                "ball_pos": (bx, by),
            }
        )

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

        # DEBUG LOG - sadece debug level'da göster
        logger.debug(
            f"Mesafe: {nearest_dist:.1f}px, Threshold: {HOLDING_THRESHOLD}px, "
            f"Y-ratio: {y_ratio:.2f}, Factor: {perspective_factor:.2f}"
        )

        # Top oyuncuya yakınsa
        if nearest_dist < HOLDING_THRESHOLD:
            if not self.ball_with_player:
                self.shooter_id = nearest[2] if len(nearest) > 2 else None
                logger.debug(f"Top yakalandı: P{self.shooter_id}")

            self.ball_with_player = True
            self.release_detected = False
            return None

        # Top oyuncudan uzaklaşıyorsa - RELEASE!
        if self.ball_with_player and not self.release_detected:
            release_info = self._check_release_condition(
                frame,
                frame_count,
                player_to_hoop_dist,
                HOLDING_THRESHOLD,
                RELEASE_THRESHOLD,
                y_ratio,
                perspective_factor,
                depth_zone,
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
        if y_ratio > 0.7:
            return 1.25, "ÖN"
        elif y_ratio > 0.5:
            return 1.1, "ORTA-ÖN"
        elif y_ratio > 0.3:
            return 1.0, "ORTA"
        else:
            return 0.85, "ARKA"

    def _calculate_dynamic_thresholds(self, player_to_hoop_dist, perspective_factor, y_ratio):
        """
        Dinamik threshold'ları hesapla.

        Returns:
            tuple: (HOLDING_THRESHOLD, RELEASE_THRESHOLD)
        """
        if player_to_hoop_dist > 300:
            holding = 70
            release = 15
        elif player_to_hoop_dist > 200:
            holding = 55
            release = 20
        elif player_to_hoop_dist > 100:
            holding = 45
            release = 25
        else:
            holding = 35
            release = 30

        holding = int(holding * perspective_factor * perspective_factor)

        if y_ratio > 0.95:
            holding = int(holding * 2.0)
        elif y_ratio > 0.85:
            holding = int(holding * 1.5)

        release = int(release / perspective_factor)

        return holding, release

    def _check_release_condition(
        self,
        frame,
        frame_count,
        player_to_hoop_dist,
        HOLDING_THRESHOLD,
        RELEASE_THRESHOLD,
        y_ratio,
        perspective_factor,
        depth_zone,
    ):
        """
        Release condition kontrolü - topun gerçekten elden çıktığını doğrula.

        Returns:
            dict veya None: Release bilgileri
        """
        if len(self.ball_player_history) < 2:
            return None

        # Son frame'leri al (deque'den liste olarak)
        history_list = list(self.ball_player_history)
        recent = history_list[-2:]

        # Mesafe artışı
        dist_increase = recent[-1]["distance"] - recent[0]["distance"]

        # Y hareketi (yukarı = negatif)
        y_movement = recent[-1]["ball_pos"][1] - recent[0]["ball_pos"][1]

        # Hız hesapla
        ball_velocity = math.sqrt(
            (recent[-1]["ball_pos"][0] - recent[0]["ball_pos"][0]) ** 2
            + (recent[-1]["ball_pos"][1] - recent[0]["ball_pos"][1]) ** 2
        )

        # Release kriterleri
        is_close_to_camera = y_ratio > 0.7

        if is_close_to_camera:
            release_condition = (
                dist_increase > RELEASE_THRESHOLD * 0.8
                or (y_movement < -3 and ball_velocity > 6)
                or ball_velocity > 10
            )
        elif player_to_hoop_dist < 100:
            release_condition = (
                dist_increase > RELEASE_THRESHOLD
                or (y_movement < -5 and ball_velocity > 8)
                or ball_velocity > 12
            )
        elif player_to_hoop_dist < 200:
            release_condition = (
                dist_increase > RELEASE_THRESHOLD
                or (y_movement < -8 and ball_velocity > 12)
                or ball_velocity > 18
            )
        else:
            release_condition = (
                dist_increase > RELEASE_THRESHOLD
                or (y_movement < -10 and ball_velocity > 15)
                or ball_velocity > 25
            )

        if not release_condition:
            return None

        # RELEASE TESPİT EDİLDİ!
        self.release_detected = True
        self.release_frame = frame_count

        # Shooter pozisyonunu bul
        release_pos, shooter_frames = self._find_shooter_position(HOLDING_THRESHOLD)
        self.release_player_pos = release_pos

        # Shot type belirle
        if player_to_hoop_dist > 300:
            shot_type = "UZAK (3PT)"
        elif player_to_hoop_dist > 200:
            shot_type = "ORTA"
        elif player_to_hoop_dist > 100:
            shot_type = "YAKIN"
        else:
            shot_type = "PAINT"

        # Log - sadece önemli eventlerde
        logger.info(f"🏀 {shot_type} ŞUT! Oyuncu: P{self.shooter_id}, Frame: {frame_count}")

        # Frame'de görsel işaretleme
        self._draw_release_marker(
            frame, shot_type, depth_zone, HOLDING_THRESHOLD, RELEASE_THRESHOLD, shooter_frames
        )

        return {
            "frame": frame_count,
            "position": self.release_player_pos,
            "shooter_id": self.shooter_id,
            "shot_type": shot_type,
            "verified": len(shooter_frames) > 0,
        }

    def _find_shooter_position(self, HOLDING_THRESHOLD):
        """
        Shooter'ın pozisyonunu history'den bul.

        Returns:
            tuple: (position, shooter_frames)
        """
        best_idx = -1
        shooter_frames = []
        history_list = list(self.ball_player_history)

        for i in range(len(history_list) - 1, -1, -1):
            frame_data = history_list[i]

            if frame_data.get("player_id") == self.shooter_id:
                if frame_data["distance"] < HOLDING_THRESHOLD * 1.3:
                    shooter_frames.append(i)
                    if best_idx == -1:
                        best_idx = i

            if len(shooter_frames) >= 5:
                break

        if best_idx == -1:
            for i in range(len(history_list) - 1, max(0, len(history_list) - 4), -1):
                if history_list[i]["distance"] < HOLDING_THRESHOLD * 1.2:
                    best_idx = i
                    break

        if best_idx != -1:
            position = history_list[best_idx]["player_pos"]
            confirmed_id = history_list[best_idx]["player_id"]
            if confirmed_id is not None:
                self.shooter_id = confirmed_id
        else:
            position = history_list[-1]["player_pos"]

        return position, shooter_frames

    def _draw_release_marker(
        self, frame, shot_type, depth_zone, HOLDING_THRESHOLD, RELEASE_THRESHOLD, shooter_frames
    ):
        """Frame üzerine release marker'ı çiz"""
        if not self.release_player_pos:
            return

        x, y = self.release_player_pos

        cv2.circle(frame, (int(x), int(y)), 15, (255, 0, 255), 3)

        cv2.putText(
            frame,
            f"P{self.shooter_id} RELEASE ({shot_type})",
            (int(x) - 80, int(y) - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
        )

        cv2.putText(
            frame,
            f"{depth_zone} | H:{HOLDING_THRESHOLD} R:{RELEASE_THRESHOLD}",
            (int(x) - 80, int(y) - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
        )

        if len(shooter_frames) > 0:
            cv2.putText(
                frame,
                "VERIFIED",
                (int(x) - 30, int(y) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )

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
            
            logger.info(f"🏀 ŞUT! Made: {made}, Shooter: P{self.shooter_id}")

            # Minimap için shot pozisyonunu kaydet
            shot_data = self._record_shot_for_minimap(
                made, H, use_flip, h_img, players, hoop_pos
            )

            # Overlay
            if made:
                self.makes += 1
                points_text = "Score"
                if shot_data and "points" in shot_data:
                    points_text = f"{shot_data['points']} PT"
                self.overlay_text = points_text
                self.overlay_color = (0, 255, 0)
            else:
                self.overlay_text = "Miss"
                self.overlay_color = (0, 0, 255)

            self.fade_counter = self.fade_frames

            # Reset
            self.up = False
            self.down = False
            self.ball_with_player = False
            self.release_detected = False
            self.release_player_pos = None
            self.shooter_id = None
            self.ball_player_history.clear()

            return shot_data

        return None

    def _record_shot_for_minimap(self, made, H, use_flip, h_img, players, hoop_pos):
        """Şutu minimap için kaydet - HER ŞUT İÇİN SKOR GÜNCELLENİR"""
        
        # Shooter ID yoksa, en yakın oyuncuyu bul
        effective_shooter_id = self.shooter_id
        if effective_shooter_id is None and players:
            # İlk oyuncuyu varsayılan olarak kullan
            if len(players) > 0 and len(players[0]) > 2:
                effective_shooter_id = players[0][2]
                logger.debug(f"Shooter ID yok, varsayılan oyuncu: P{effective_shooter_id}")
        
        # Varsayılan puan değeri
        points_val = 2
        mx, my = 0, 0
        
        # Release pozisyonu varsa kullan
        if self.release_player_pos and effective_shooter_id is not None:
            try:
                pt = np.array(
                    [[[self.release_player_pos[0], self.release_player_pos[1]]]],
                    dtype=np.float32,
                )
                proj_pt = cv2.perspectiveTransform(pt, H)[0][0]
                mx = int(proj_pt[0])
                my = int(proj_pt[1])

                if use_flip:
                    my = h_img - my

                points_val = self._classify_shot_2pt_or_3pt(mx, my)

                self.shot_history.append((mx, my, made, effective_shooter_id, points_val))
                # Memory leak önleme - eski kayıtları sil
                if len(self.shot_history) > self.MAX_SHOT_HISTORY:
                    self.shot_history = self.shot_history[-self.MAX_SHOT_HISTORY:]
                
                self._update_player_score(effective_shooter_id, made, points_val)
                
                logger.info(
                    f"📍 Minimap: ({mx}, {my}), P{effective_shooter_id}, "
                    f"{'✓' if made else '✗'} {points_val}PT"
                )

                return {
                    "minimap_pos": (mx, my),
                    "made": made,
                    "shooter_id": effective_shooter_id,
                    "points": points_val,
                }
            except Exception as e:
                logger.debug(f"Release pos transform hatası: {e}")

        # Fallback: Oyuncu pozisyonundan hesapla
        if effective_shooter_id is not None and players:
            for p in players:
                if len(p) > 2 and p[2] == effective_shooter_id:
                    try:
                        pt = np.array([[[p[0], p[1]]]], dtype=np.float32)
                        proj_pt = cv2.perspectiveTransform(pt, H)[0][0]
                        mx = int(proj_pt[0])
                        my = int(proj_pt[1])

                        if use_flip:
                            my = h_img - my

                        points_val = self._classify_shot_2pt_or_3pt(mx, my)

                        self.shot_history.append((mx, my, made, effective_shooter_id, points_val))
                        # Memory leak önleme
                        if len(self.shot_history) > self.MAX_SHOT_HISTORY:
                            self.shot_history = self.shot_history[-self.MAX_SHOT_HISTORY:]
                        
                        self._update_player_score(effective_shooter_id, made, points_val)
                        
                        logger.info(
                            f"📍 Minimap (fallback): ({mx}, {my}), P{effective_shooter_id}, "
                            f"{'✓' if made else '✗'} {points_val}PT"
                        )

                        return {
                            "minimap_pos": (mx, my),
                            "made": made,
                            "shooter_id": effective_shooter_id,
                            "points": points_val,
                        }
                    except Exception:
                        continue
        
        # Son fallback: Oyuncu bilinmese bile skoru güncelle
        if effective_shooter_id is not None:
            self._update_player_score(effective_shooter_id, made, points_val)
            logger.info(f"📍 Skor güncellendi (no minimap): P{effective_shooter_id}, {'✓' if made else '✗'} {points_val}PT")
            return {
                "minimap_pos": None,
                "made": made,
                "shooter_id": effective_shooter_id,
                "points": points_val,
            }
        
        # Hiç oyuncu yoksa bile default oyuncu (P1) için skor güncelle
        default_player = 1
        self._update_player_score(default_player, made, points_val)
        logger.info(f"📍 Skor güncellendi (default): P{default_player}, {'✓' if made else '✗'} {points_val}PT")
        return {
            "minimap_pos": None,
            "made": made,
            "shooter_id": default_player,
            "points": points_val,
        }

    def _classify_shot_2pt_or_3pt(self, mx, my):
        """
        Minimap koordinatlarına göre 2PT/3PT sınıflandırma.
        OPTIMIZED: Cache'lenmiş polygon kullanır.
        
        Args:
            mx, my: Minimap üzerindeki şut pozisyonu
            
        Returns:
            int: 2 veya 3
        """
        if self._three_point_polygon is None:
            return 2  # Varsayılan
        
        try:
            result = cv2.pointPolygonTest(
                self._three_point_polygon, (float(mx), float(my)), False
            )
            return 2 if result >= 0 else 3
        except Exception:
            return 2

    def _update_player_score(self, player_id, made, points_val):
        """Oyuncu skorunu güncelle."""
        if player_id is None:
            return
            
        if player_id not in self.player_scores:
            self.player_scores[player_id] = {
                "points": 0,
                "makes": 0,
                "attempts": 0
            }
        
        self.player_scores[player_id]["attempts"] += 1
        
        if made:
            self.player_scores[player_id]["points"] += points_val
            self.player_scores[player_id]["makes"] += 1
            
        stats = self.player_scores[player_id]
        logger.info(f"🏀 P{player_id}: {stats['points']}pts ({stats['makes']}/{stats['attempts']})")

    def update_fade(self):
        """Fade counter'ı güncelle"""
        if self.fade_counter > 0:
            self.fade_counter -= 1

    def get_stats(self):
        """İstatistikleri döndür"""
        return {
            "makes": self.makes,
            "attempts": self.attempts,
            "percentage": (self.makes / self.attempts * 100) if self.attempts > 0 else 0,
        }
    
    def get_player_scores(self):
        """Oyuncu skorlarını döndür"""
        return self.player_scores
