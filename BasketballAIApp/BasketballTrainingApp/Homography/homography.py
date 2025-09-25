import cv2
import numpy as np
import json
import os
import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO

# ------------------------ CONFIG ------------------------
court_labels = [
    "top_left", "top_right", "bottom_right", "bottom_left",
    "free_throw_left", "free_throw_right",
    "center_circle", "paint_left", "paint_right"
]

video_path = r"D:\repos\Basketball_App\BasketballAIApp\clips\training7.mp4"
minimap_path = r"D:\repos\Basketball_App\BasketballAIApp\BasketballTrainingApp\Homography\images\hom.png"
yolo_model_path = r"D:\repos\Basketball_App\yolov8s.pt"

vp_json = "video_points.json"
mp_json = "minimap_points.json"
scale = 0.5  # video resize
max_jump = 30  # ani değişim için maksimum piksel
# --------------------------------------------------------

# ------------------------ UI ------------------------
def ask_label():
    root = tk.Tk()
    root.title("Nokta Seçimi")
    var = tk.StringVar(value=court_labels[0])
    tk.Label(root, text="Bu nokta hangisi?").pack(padx=10, pady=5)
    combo = ttk.Combobox(root, values=court_labels, textvariable=var, state="readonly")
    combo.pack(padx=10, pady=5)
    tk.Button(root, text="Seç", command=root.quit).pack(pady=5)
    root.mainloop()
    choice = var.get()
    root.destroy()
    return choice

def on_mouse(event, x, y, flags, param):
    points_dict = param
    if event == cv2.EVENT_LBUTTONDOWN:
        label = ask_label()
        points_dict[label] = (int(x), int(y))
        print(f"{label} seçildi: ({x},{y})")

def select_points(image, title):
    cv2.namedWindow(title)
    points_dict = {}
    cv2.setMouseCallback(title, on_mouse, points_dict)
    print(f"{title} üzerinde noktaları tıkla ve label seç. ESC ile bitir.")
    while True:
        tmp = image.copy()
        for label, pt in points_dict.items():
            cv2.circle(tmp, pt, 6, (0, 0, 255), -1)
            cv2.putText(tmp, label, (pt[0]+6, pt[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow(title, tmp)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyWindow(title)
    return points_dict

# ------------------------ HOMOGRAPHY ------------------------
def compute_h_and_error(video_pts, minimap_pts):
    H, mask = cv2.findHomography(video_pts, minimap_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None, float('inf')
    proj = cv2.perspectiveTransform(video_pts.reshape(-1,1,2), H).reshape(-1,2)
    errs = np.linalg.norm(proj - minimap_pts, axis=1)
    return H, float(np.mean(errs)), proj, errs

def load_or_select(video_frame, minimap_img, vp_json, mp_json):
    if os.path.exists(vp_json) and os.path.exists(mp_json):
        with open(vp_json, "r") as f:
            video_points_dict = json.load(f)
        with open(mp_json, "r") as f:
            minimap_points_dict = json.load(f)
    else:
        video_points_dict = select_points(video_frame, "Video")
        minimap_points_dict = select_points(minimap_img, "Minimap")
        with open(vp_json, "w") as f:
            json.dump(video_points_dict, f, indent=2)
        with open(mp_json, "w") as f:
            json.dump(minimap_points_dict, f, indent=2)
        print("Noktalar kaydedildi.")
    video_points_dict = {k: (int(v[0]), int(v[1])) for k,v in video_points_dict.items()}
    minimap_points_dict = {k: (int(v[0]), int(v[1])) for k,v in minimap_points_dict.items()}
    return video_points_dict, minimap_points_dict

# ------------------------ MAIN ------------------------
def main():
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("Video okunamadı")
        return
    first_frame = cv2.resize(first_frame, (int(first_frame.shape[1]*scale), int(first_frame.shape[0]*scale)))

    minimap_img = cv2.imread(minimap_path)
    if minimap_img is None:
        print("Minimap bulunamadı!")
        return

    video_points_dict, minimap_points_dict = load_or_select(first_frame, minimap_img, vp_json, mp_json)

    common_labels = [l for l in court_labels if l in video_points_dict and l in minimap_points_dict]
    if len(common_labels) < 4:
        print("Homography için en az 4 ortak nokta gerekli!", common_labels)
        return
    print("Ortak etiketler:", common_labels)

    video_pts = np.array([video_points_dict[l] for l in common_labels], dtype=np.float32)
    minimap_pts = np.array([minimap_points_dict[l] for l in common_labels], dtype=np.float32)

    H1, err1, proj1, errs1 = compute_h_and_error(video_pts, minimap_pts)
    h_img = minimap_img.shape[0]
    minimap_pts_flipped = np.array([[x, h_img - y] for (x,y) in minimap_pts], dtype=np.float32)
    H2, err2, proj2, errs2 = compute_h_and_error(video_pts, minimap_pts_flipped)

    use_flip = False
    H, proj, errs, err = H1, proj1, errs1, err1
    if H1 is None and H2 is None:
        print("Homography hesaplanamadı")
        return
    elif H1 is None or (err2 + 1.0 < err1):
        use_flip = True
        H, proj, errs, err = H2, proj2, errs2, err2

    print(f"Seçilen dönüşüm: flip_y={use_flip}, ortalama reproj hatası={err:.2f}")

    # ------------------------ YOLO MODEL ------------------------
    model = YOLO(yolo_model_path)

    # ------------------------ CANLI VIDEO + MINIMAP ------------------------
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_players = []  # önceki frame konumları

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))

        results = model(frame)[0]
        players = []

        for i, box in enumerate(results.boxes):
            cls_index = int(box.cls.cpu().numpy())
            cls_name = results.names[cls_index]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ayak kısmını al
            cx = (x1 + x2)//2
            cy = y2

            # --- DEBUG ---
            print(f"[DEBUG] Box {i}: {cls_name} video=({x1},{y1})-({x2},{y2}), foot=({cx},{cy})")

            if cls_name == "person":
                # ani değişimi filtrele
                if i < len(prev_players):
                    px, py = prev_players[i]
                    dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                    if dist > max_jump:
                        cx, cy = px, py  # ani sıçramayı yansıtma
                players.append((cx, cy))
                cv2.circle(frame, (cx, cy), 5, (0,255,255), -1)

        prev_players = players.copy()  # bu frame konumlarını kaydet

        # --- Minimap AR ---
        minimap_copy = minimap_img.copy()
        for i,l in enumerate(common_labels):
            tx, ty = minimap_pts[i]
            cv2.circle(minimap_copy, (int(tx), int(ty)), 5, (0,255,0), -1)

        for cx, cy in players:
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            proj_pt = cv2.perspectiveTransform(pt, H)[0][0]
            display_x = int(proj_pt[0])
            display_y = int(proj_pt[1])
            if use_flip:
                display_y = h_img - display_y
            inside = (0 <= display_x < minimap_copy.shape[1] and 0 <= display_y < minimap_copy.shape[0])
            color = (0,0,255) if inside else (0,0,128)
            cv2.circle(minimap_copy, (np.clip(display_x,0,minimap_copy.shape[1]-1),
                                      np.clip(display_y,0,minimap_copy.shape[0]-1)), 8, color, -1)
            if not inside:
                cv2.putText(minimap_copy, "OUT", (np.clip(display_x,0,minimap_copy.shape[1]-1)+6,
                                                  np.clip(display_y,0,minimap_copy.shape[0]-1)-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.imshow("Video", frame)
        cv2.imshow("Minimap AR", minimap_copy)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
