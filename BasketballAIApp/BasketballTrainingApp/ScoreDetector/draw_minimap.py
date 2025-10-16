
import cv2
import numpy as np

def draw_minimap(minimap_img,shot_history, players,H,use_flip,h_img):
    minimap_copy = minimap_img.copy()

    # Şut pozisyonlarını işaretle (RELEASE POINT'ler)
    for shot_data in shot_history:
        mx = shot_data[0]
        my = shot_data[1]
        made = shot_data[2]
        shooter_id = None
        points_val = None
        if len(shot_data) >= 4:
            shooter_id = shot_data[3]
        if len(shot_data) >= 5:
            points_val = shot_data[4]

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

    # Oyuncular (cx,cy,ID)
    for cx, cy, pid in players:
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        proj_pt = cv2.perspectiveTransform(pt, H)[0][0]
        display_x = int(proj_pt[0])
        display_y = int(proj_pt[1])
        if use_flip:
            display_y = h_img - display_y
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
    
    # Return the minimap for display
    return minimap_copy