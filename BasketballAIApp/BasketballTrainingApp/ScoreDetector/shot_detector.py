"""
Shot Detection Module
=====================
Shot tespiti ve release point detection için modüler class.
tracking.py'den ayrıştırıldı - daha temiz ve maintainable kod.
"""

import cv2
import math
import numpy as np
import json
import os
from utilsfixed import score, detect_down, detect_up


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
        
        # Player scores tracking
        self.player_scores = {}  # {player_id: {"points": total_points, "makes": count, "attempts": count}}

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
        nearest = min(
            players, key=lambda p: math.sqrt((p[0] - bx) ** 2 + (p[1] - by) ** 2)
        )
        nearest_dist = math.sqrt((nearest[0] - bx) ** 2 + (nearest[1] - by) ** 2)
        px, py = nearest[0], nearest[1]

        # Top-oyuncu mesafe geçmişini kaydet
        self.ball_player_history.append(
            {
                "frame": frame_count,
                "distance": nearest_dist,
                "player_pos": (px, py),
                "player_id": nearest[2] if len(nearest) > 2 else None,
                "ball_pos": (bx, by),
            }
        )

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
        print(
            f"[DEBUG] Mesafe: {nearest_dist:.1f}px, Threshold: {HOLDING_THRESHOLD}px, "
            f"Y-ratio: {y_ratio:.2f}, Factor: {perspective_factor:.2f}, "
            f"Hoop dist: {player_to_hoop_dist:.0f}px"
        )

        # Top oyuncuya yakınsa
        if nearest_dist < HOLDING_THRESHOLD:
            if not self.ball_with_player:
                self.shooter_id = nearest[2] if len(nearest) > 2 else None
                print(
                    f"   ✅ Top yakalandı: Oyuncu P{self.shooter_id} (mesafe: {nearest_dist:.1f}px < {HOLDING_THRESHOLD}px)"
                )

            self.ball_with_player = True
            self.release_detected = False
            return None
        else:
            # DEBUG: Neden yakalanmadı?
            if not self.ball_with_player:
                print(
                    f"   ❌ Top yakalanmadı: {nearest_dist:.1f}px >= {HOLDING_THRESHOLD}px (fark: {nearest_dist - HOLDING_THRESHOLD:.1f}px)"
                )

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
        Dinamik threshold'ları hesapla - KAMERAYA YAKIN İÇİN ÇOK YÜKSEK!

        Args:
            player_to_hoop_dist: Oyuncunun potaya mesafesi
            perspective_factor: Perspektif faktörü
            y_ratio: Oyuncunun frame'deki dikey pozisyonu (0=üst, 1=alt)

        Returns:
            tuple: (HOLDING_THRESHOLD, RELEASE_THRESHOLD)
        """
        if player_to_hoop_dist > 300:  # Uzak şut (3-point)
            holding = 70
            release = 15
        elif player_to_hoop_dist > 200:  # Orta mesafe
            holding = 55
            release = 20
        elif player_to_hoop_dist > 100:  # Yakın-orta
            holding = 45
            release = 25
        else:  # ÇOK YAKIN şut (paint area)
            holding = 35
            release = 30

        # Perspektif faktörünü uygula - KAMERAYA YAKIN = ÇOK DAHA YÜKSEK THRESHOLD!
        # 3D'de yakın top, 2D projeksiyonda çok uzak görünür (150px+)
        holding = int(holding * perspective_factor * perspective_factor)  # Kare al (1.25^2 = 1.56)

        # EKSTRA BOOST: Kameraya ÇOK yakın oyuncular için (y_ratio > 0.95)
        # Senin durumunda: y_ratio=0.99-1.00, mesafe=150px
        if y_ratio > 0.95:
            holding = int(holding * 2.0)  # 2x ek artış! (150px+ için)
        elif y_ratio > 0.85:
            holding = int(holding * 1.5)  # 1.5x ek artış

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

        # Son frame'leri al
        recent = self.ball_player_history[-2:]

        # Mesafe artışı
        dist_increase = recent[-1]["distance"] - recent[0]["distance"]

        # Y hareketi (yukarı = negatif)
        y_movement = recent[-1]["ball_pos"][1] - recent[0]["ball_pos"][1]

        # Hız hesapla
        ball_velocity = math.sqrt(
            (recent[-1]["ball_pos"][0] - recent[0]["ball_pos"][0]) ** 2
            + (recent[-1]["ball_pos"][1] - recent[0]["ball_pos"][1]) ** 2
        )

        # Release kriterleri - MESAFE VE KAMERAYA YAKINLIK İÇİN ADAPTIF
        # Kameraya çok yakın pozisyonlarda (y_ratio > 0.7) daha esnek ol
        is_close_to_camera = y_ratio > 0.7

        if is_close_to_camera:
            # KAMERAYA YAKIN - Perspektif etkisi nedeniyle çok esnek kriterler
            release_condition = (
                dist_increase > RELEASE_THRESHOLD * 0.8
                or (y_movement < -3 and ball_velocity > 6)
                or ball_velocity > 10
            )
        elif player_to_hoop_dist < 100:  # ÇOK YAKIN ŞUT (ama kameraya uzak)
            release_condition = (
                dist_increase > RELEASE_THRESHOLD
                or (y_movement < -5 and ball_velocity > 8)
                or ball_velocity > 12
            )
        elif player_to_hoop_dist < 200:  # YAKIN-ORTA
            release_condition = (
                dist_increase > RELEASE_THRESHOLD
                or (y_movement < -8 and ball_velocity > 12)
                or ball_velocity > 18
            )
        else:  # UZAK ŞUT
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
            frame_count,
            shot_type,
            depth_zone,
            y_ratio,
            dist_increase,
            ball_velocity,
            player_to_hoop_dist,
            HOLDING_THRESHOLD,
            RELEASE_THRESHOLD,
            perspective_factor,
            shooter_frames,
        )

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

        # Geriye doğru git ve shooter_id'ye sahip frame'leri bul
        for i in range(len(self.ball_player_history) - 1, -1, -1):
            frame_data = self.ball_player_history[i]

            if frame_data.get("player_id") == self.shooter_id:
                if frame_data["distance"] < HOLDING_THRESHOLD * 1.3:
                    shooter_frames.append(i)
                    if best_idx == -1:
                        best_idx = i

            # Son 8 frame yeterli
            if len(shooter_frames) >= 5:
                break

        # Fallback: Eğer shooter bulunamazsa
        if best_idx == -1:
            for i in range(len(self.ball_player_history) - 1, max(0, len(self.ball_player_history) - 4), -1):
                if self.ball_player_history[i]["distance"] < HOLDING_THRESHOLD * 1.2:
                    best_idx = i
                    break

        # Pozisyonu döndür
        if best_idx != -1:
            position = self.ball_player_history[best_idx]["player_pos"]
            # Shooter ID'yi doğrula
            confirmed_id = self.ball_player_history[best_idx]["player_id"]
            if confirmed_id is not None:
                self.shooter_id = confirmed_id
        else:
            # En son pozisyon
            position = self.ball_player_history[-1]["player_pos"]

        return position, shooter_frames

    def _log_release_info(
        self,
        frame_count,
        shot_type,
        depth_zone,
        y_ratio,
        dist_increase,
        ball_velocity,
        player_to_hoop_dist,
        HOLDING_THRESHOLD,
        RELEASE_THRESHOLD,
        perspective_factor,
        shooter_frames,
    ):
        """Release bilgilerini logla"""
        camera_proximity = "KAMERAYA YAKIN" if y_ratio > 0.7 else ""
        print(f"🏀 {shot_type} ŞUT ATILDI! [Derinlik: {depth_zone}] {camera_proximity}")
        print(
            f"   Frame: {frame_count}, Oyuncu: P{self.shooter_id} "
            f"({'✓ Doğrulandı' if len(shooter_frames) > 0 else '⚠ Fallback'})"
        )
        print(f"   Pozisyon: {self.release_player_pos}, Y-oranı: {y_ratio:.2f}")
        print(f"   Mesafe artışı: {dist_increase:.1f}px, Hız: {ball_velocity:.1f}px/f")
        print(f"   Potaya mesafe: {player_to_hoop_dist:.0f}px")
        print(
            f"   Threshold: HOLDING={HOLDING_THRESHOLD}px, "
            f"RELEASE={RELEASE_THRESHOLD}px (faktör={perspective_factor:.2f})"
        )
        print(f"   Shooter frame sayısı: {len(shooter_frames)} (aynı oyuncuya ait)")

    def _draw_release_marker(
        self, frame, shot_type, depth_zone, HOLDING_THRESHOLD, RELEASE_THRESHOLD, shooter_frames
    ):
        """Frame üzerine release marker'ı çiz"""
        if not self.release_player_pos:
            return

        x, y = self.release_player_pos

        # Ana işaret
        cv2.circle(frame, (int(x), int(y)), 15, (255, 0, 255), 3)

        # Oyuncu ID ve shot type
        cv2.putText(
            frame,
            f"P{self.shooter_id} RELEASE ({shot_type})",
            (int(x) - 80, int(y) - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
        )

        # Zone ve threshold bilgisi
        cv2.putText(
            frame,
            f"{depth_zone} | H:{HOLDING_THRESHOLD} R:{RELEASE_THRESHOLD}",
            (int(x) - 80, int(y) - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
        )

        # Doğrulama işareti
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

            # Minimap için shot pozisyonunu kaydet (2PT/3PT hesapla)
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
            self.ball_player_history = []

            return shot_data

        return None

    def _record_shot_for_minimap(self, made, H, use_flip, h_img, players, hoop_pos):
        """Şutu minimap için kaydet"""
        # Primary: release pozisyonunu kullan
        if self.release_player_pos and self.shooter_id is not None:
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

                # 3PT sınıflandırma - JSON'dan poligon oku
                points_val = self._classify_shot_2pt_or_3pt(mx, my)

                self.shot_history.append((mx, my, made, self.shooter_id, points_val))
                
                # Oyuncu skorunu güncelle
                self._update_player_score(self.shooter_id, made, points_val)
                
                print(
                    f"📍 Minimap'e eklendi: ({mx}, {my}), Oyuncu: P{self.shooter_id}, "
                    f"{'BAŞARILI' if made else 'BAŞARISIZ'} | {points_val}PT"
                )

                return {
                    "minimap_pos": (mx, my),
                    "made": made,
                    "shooter_id": self.shooter_id,
                    "points": points_val,
                }
            except Exception:
                # Fallthrough to fallback below
                pass

        # Fallback: Mevcut oyuncu pozisyonunu kullan
        if self.shooter_id is not None and players:
            for p in players:
                if len(p) > 2 and p[2] == self.shooter_id:
                    try:
                        pt = np.array([[[p[0], p[1]]]], dtype=np.float32)
                        proj_pt = cv2.perspectiveTransform(pt, H)[0][0]
                        mx = int(proj_pt[0])
                        my = int(proj_pt[1])

                        if use_flip:
                            my = h_img - my

                        # 3PT sınıflandırma
                        points_val = self._classify_shot_2pt_or_3pt(mx, my)

                        self.shot_history.append((mx, my, made, self.shooter_id, points_val))
                        
                        # Oyuncu skorunu güncelle
                        self._update_player_score(self.shooter_id, made, points_val)
                        
                        print(
                            f"📍 Minimap'e eklendi (FALLBACK): ({mx}, {my}), Oyuncu: P{self.shooter_id} | {points_val}PT"
                        )

                        return {
                            "minimap_pos": (mx, my),
                            "made": made,
                            "shooter_id": self.shooter_id,
                            "points": points_val,
                        }
                    except Exception:
                        continue

        return None

    def _classify_shot_2pt_or_3pt(self, mx, my):
        """
        Minimap koordinatlarına göre 2PT/3PT sınıflandırma.
        
        Args:
            mx, my: Minimap üzerindeki şut pozisyonu
            
        Returns:
            int: 2 veya 3
        """
        try:
            # three_point_line.json'u oku
            base_dir = os.path.dirname(__file__)
            json_path = os.path.join(base_dir, "three_point_line.json")
            
            if not os.path.exists(json_path):
                print(f"⚠️  {json_path} bulunamadı! Önce select_3pt_line.py'ı çalıştırın.")
                return 2  # Varsayılan
            
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            points_3pt = data.get("points", [])
            if len(points_3pt) < 3:
                print(f"⚠️  3PT çizgisi yetersiz nokta içeriyor: {len(points_3pt)}")
                return 2
            
            # Point-in-polygon test (OpenCV)
            polygon = np.array(points_3pt, dtype=np.float32)
            result = cv2.pointPolygonTest(polygon, (float(mx), float(my)), False)
            
            # result > 0: İçeride (2PT)
            # result < 0: Dışarıda (3PT)
            # result = 0: Çizgi üzerinde (2PT kabul edelim)
            
            points_val = 2 if result >= 0 else 3
            
            return points_val
            
        except Exception as e:
            print(f"⚠️  3PT sınıflandırma hatası: {e}")
            return 2  # Varsayılan

    def _update_player_score(self, player_id, made, points_val):
        """
        Oyuncu skorunu güncelle.
        
        Args:
            player_id: Oyuncu ID'si
            made: Şut başarılı mı? (True/False)
            points_val: Kazanılan puan (2 veya 3)
        """
        if player_id is None:
            return
            
        # Oyuncu ilk kez görülüyorsa, kayıt oluştur
        if player_id not in self.player_scores:
            self.player_scores[player_id] = {
                "points": 0,
                "makes": 0,
                "attempts": 0
            }
        
        # Deneme sayısını artır
        self.player_scores[player_id]["attempts"] += 1
        
        # Başarılı şutta puan ekle
        if made:
            self.player_scores[player_id]["points"] += points_val
            self.player_scores[player_id]["makes"] += 1
            
        print(f"🏀 Oyuncu P{player_id} Skorlar: {self.player_scores[player_id]['points']} puan "
              f"({self.player_scores[player_id]['makes']}/{self.player_scores[player_id]['attempts']})")

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
