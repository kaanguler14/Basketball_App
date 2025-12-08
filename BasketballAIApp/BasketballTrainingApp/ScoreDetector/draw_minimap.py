# draw_minimap.py
# OPTIMIZED VERSION: Batch homography transformation

import cv2
import numpy as np


def draw_minimap(minimap_img, shot_history, players, H, use_flip, h_img):
    """
    Minimap çizimi - Optimized with batch homography transformation.
    
    Args:
        minimap_img: Orijinal minimap resmi
        shot_history: Şut geçmişi listesi
        players: Oyuncu listesi [(cx, cy, id), ...]
        H: Homography matrisi
        use_flip: Y eksenini çevir
        h_img: Resim yüksekliği (flip için)
    
    Returns:
        numpy.ndarray: Güncellenmiş minimap
    """
    minimap_copy = minimap_img.copy()
    minimap_h, minimap_w = minimap_copy.shape[:2]

    # Şut pozisyonlarını işaretle (RELEASE POINT'ler)
    for shot_data in shot_history:
        mx = shot_data[0]
        my = shot_data[1]
        made = shot_data[2]
        shooter_id = shot_data[3] if len(shot_data) >= 4 else None
        points_val = shot_data[4] if len(shot_data) >= 5 else None

        # Renk: Yeşil = başarılı, Kırmızı = kaçan
        color = (0, 255, 0) if made else (0, 0, 255)

        # X işareti (şut pozisyonu)
        cv2.line(minimap_copy, (mx - 10, my - 10), (mx + 10, my + 10), color, 3)
        cv2.line(minimap_copy, (mx - 10, my + 10), (mx + 10, my - 10), color, 3)

        # Daire (release point vurgusu)
        cv2.circle(minimap_copy, (mx, my), 8, color, 2)

        # Oyuncu ID/puan (varsa)
        label_parts = []
        if shooter_id is not None:
            label_parts.append(f"P{shooter_id}")
        if points_val is not None:
            label_parts.append(f"{points_val}PT")
        if label_parts:
            cv2.putText(minimap_copy, " ".join(label_parts), (mx + 12, my - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # OPTIMIZED: Batch homography transformation for players
    if players:
        # Tüm oyuncu pozisyonlarını tek bir array'de topla
        player_points = np.array([[[p[0], p[1]]] for p in players], dtype=np.float32)
        
        # Tek seferde tüm noktaları dönüştür (batch processing)
        if len(player_points) > 0:
            projected_points = cv2.perspectiveTransform(player_points, H)
            
            for i, (cx, cy, pid) in enumerate(players):
                proj_pt = projected_points[i][0]
                display_x = int(proj_pt[0])
                display_y = int(proj_pt[1])
                
                if use_flip:
                    display_y = h_img - display_y
                
                # Sınır kontrolü
                inside = (0 <= display_x < minimap_w and 0 <= display_y < minimap_h)
                color = (0, 0, 255) if inside else (0, 0, 128)
                
                # Koordinatları sınırla
                draw_x = np.clip(display_x, 0, minimap_w - 1)
                draw_y = np.clip(display_y, 0, minimap_h - 1)
                
                cv2.circle(minimap_copy, (draw_x, draw_y), 8, color, -1)
                cv2.putText(minimap_copy, f"P{pid}", (display_x + 8, display_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                if not inside:
                    cv2.putText(minimap_copy, "OUT", (draw_x + 6, draw_y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return minimap_copy
