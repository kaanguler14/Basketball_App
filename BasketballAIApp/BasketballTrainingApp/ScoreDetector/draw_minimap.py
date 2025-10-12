
import cv2
import numpy as np

def draw_minimap(minimap_img,shot_history, players,H,use_flip,h_img):
    minimap_copy = minimap_img.copy()

    # Şut pozisyonlarını işaretle (RELEASE POINT'ler)
    for shot_data in shot_history:
        if len(shot_data) == 4:
            mx, my, made, shooter_id = shot_data
        else:
            mx, my, made = shot_data
            shooter_id = None

        # Renk: Yeşil = başarılı, Kırmızı = kaçan
        color = (0, 255, 0) if made else (0, 0, 255)

        # X işareti (şut pozisyonu)
        cv2.line(minimap_copy, (mx - 10, my - 10), (mx + 10, my + 10), color, 3)
        cv2.line(minimap_copy, (mx - 10, my + 10), (mx + 10, my - 10), color, 3)

        # Daire (release point vurgusu)
        cv2.circle(minimap_copy, (mx, my), 8, color, 2)

        # Oyuncu ID'si (varsa)
        if shooter_id is not None:
            cv2.putText(minimap_copy, f"P{shooter_id}", (mx + 12, my - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

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
    cv2.imshow("Minimap AR", minimap_copy)