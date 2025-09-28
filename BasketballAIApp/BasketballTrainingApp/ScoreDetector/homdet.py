from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import json
import os
from utilsfixed import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device

# ------------------------ CONFIG ------------------------
court_labels = [
    "top_left", "top_right", "bottom_right", "bottom_left",
    "free_throw_left", "free_throw_right",
    "center_circle", "paint_left", "paint_right"
]

vp_json = "video_points.json"
mp_json = "minimap_points.json"
scale = 0.5  # video resize


class ShotDetector:
    def __init__(self):
        # --- YOLO MODELLERİ ---
        self.model_ball = YOLO("D://repos//Basketball_App//BasketballAIApp//Trainings//kagglebest.pt")
        self.model_player = YOLO("D://repos//Basketball_App//yolov8s.pt")  # person detection
        self.device = get_device()

        # --- VIDEO / MINIMAP ---
        self.cap = cv2.VideoCapture(r"D:\repos\Basketball_App\BasketballAIApp\clips\training7.mp4")
        self.minimap_img = cv2.imread(r"D:\repos\Basketball_App\BasketballAIApp\BasketballTrainingApp\Homography\images\hom.png")
        self.frame_count = 0
        self.frame = None

        # --- BALL/HOOP LOGIC ---
        self.ball_pos = []
        self.hoop_pos = []
        self.makes = 0
        self.attempts = 0
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        # --- OVERLAY ---
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_text = "waiting.."
        self.overlay_color = (0, 0, 0)

        # --- Homography ---
        ret, first_frame = self.cap.read()
        first_frame = cv2.resize(first_frame, (int(first_frame.shape[1]*scale), int(first_frame.shape[0]*scale)))
        self.video_points_dict, self.minimap_points_dict = self.load_or_select(first_frame)
        self.H, self.use_flip, self.h_img = self.compute_homography()

        self.shot_history = []  # x,y,made
        self.run()

    # ------------------------ HOMOGRAPHY ------------------------
    def load_or_select(self, video_frame):
        force_select = False
        if os.path.exists(vp_json) and os.path.exists(mp_json):
            if self.ask_reload():
                force_select = True

        if not force_select and os.path.exists(vp_json) and os.path.exists(mp_json):
            with open(vp_json, "r") as f:
                video_points_dict = json.load(f)
            with open(mp_json, "r") as f:
                minimap_points_dict = json.load(f)
        else:
            video_points_dict = self.select_points(video_frame, "Video")
            minimap_points_dict = self.select_points(self.minimap_img, "Minimap")
            with open(vp_json, "w") as f:
                json.dump(video_points_dict, f, indent=2)
            with open(mp_json, "w") as f:
                json.dump(minimap_points_dict, f, indent=2)
            print("Noktalar kaydedildi.")
        video_points_dict = {k: (int(v[0]), int(v[1])) for k, v in video_points_dict.items()}
        minimap_points_dict = {k: (int(v[0]), int(v[1])) for k, v in minimap_points_dict.items()}
        return video_points_dict, minimap_points_dict

    def ask_reload(self):
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        result = messagebox.askyesno("JSON bulundu", "Önceden kaydedilmiş noktalar bulundu.\nYeniden seçmek ister misin?")
        root.destroy()
        return result

    def select_points(self, image, title):
        import tkinter as tk
        from tkinter import ttk
        points_dict = {}
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                root = tk.Tk()
                root.title("Nokta Seçimi")
                var = tk.StringVar(value=court_labels[0])
                tk.Label(root, text="Bu nokta hangisi?").pack(padx=10, pady=5)
                combo = ttk.Combobox(root, values=court_labels, textvariable=var, state="readonly")
                combo.pack(padx=10, pady=5)
                tk.Button(root, text="Seç", command=root.quit).pack(pady=5)
                root.mainloop()
                label = var.get()
                root.destroy()
                points_dict[label] = (int(x), int(y))
                print(f"{label} seçildi: ({x},{y})")
        cv2.namedWindow(title)
        cv2.setMouseCallback(title, on_mouse)
        while True:
            tmp = image.copy()
            for label, pt in points_dict.items():
                cv2.circle(tmp, pt, 6, (0, 0, 255), -1)
                cv2.putText(tmp, label, (pt[0]+6, pt[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cv2.imshow(title, tmp)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyWindow(title)
        return points_dict

    def compute_homography(self):
        common_labels = [l for l in court_labels if l in self.video_points_dict and l in self.minimap_points_dict]
        if len(common_labels) < 4:
            raise ValueError("Homography için en az 4 ortak nokta gerekli!")
        video_pts = np.array([self.video_points_dict[l] for l in common_labels], dtype=np.float32)
        minimap_pts = np.array([self.minimap_points_dict[l] for l in common_labels], dtype=np.float32)

        H1, err1, proj1, errs1 = self.compute_h_and_error(video_pts, minimap_pts)
        h_img = self.minimap_img.shape[0]
        minimap_pts_flipped = np.array([[x, h_img - y] for (x, y) in minimap_pts], dtype=np.float32)
        H2, err2, proj2, errs2 = self.compute_h_and_error(video_pts, minimap_pts_flipped)

        use_flip = False
        H, proj, errs, err = H1, proj1, errs1, err1
        if H1 is None and H2 is None:
            raise ValueError("Homography hesaplanamadı")
        elif H1 is None or (err2 + 1.0 < err1):
            use_flip = True
            H, proj, errs, err = H2, proj2, errs2, err2
        print(f"Seçilen dönüşüm: flip_y={use_flip}, ortalama reproj hatası={err:.2f}")
        return H, use_flip, h_img

    def compute_h_and_error(self, video_pts, minimap_pts):
        H, mask = cv2.findHomography(video_pts, minimap_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None, float('inf'), None, None
        proj = cv2.perspectiveTransform(video_pts.reshape(-1,1,2), H).reshape(-1,2)
        errs = np.linalg.norm(proj - minimap_pts, axis=1)
        return H, float(np.mean(errs)), proj, errs

    # ------------------------ RUN ------------------------
    def run(self):
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break
            self.frame = cv2.resize(self.frame, (int(self.frame.shape[1]*scale), int(self.frame.shape[0]*scale)))

            # --- BALL + HOOP ---
            results_ball = self.model_ball(self.frame, stream=True, device=self.device)
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

            # --- PLAYER DETECTION ---
            results_player = self.model_player(self.frame, stream=True, device=self.device)
            players = []
            for r in results_player:
                for box in r.boxes:
                    cls_index = int(box.cls.cpu().numpy())
                    cls_name = r.names[cls_index]
                    if cls_name != "person":
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1+x2)//2, y2
                    players.append((cx, cy))
                    cv2.circle(self.frame, (cx, cy), 5, (0, 255, 255), -1)

            # --- BALL & HOOP CLEANING + SHOT DETECTION ---
            self.clean_motion()
            self.shot_detection()

            # --- MINIMAP ---
            self.draw_minimap(players)

            # --- SCORE OVERLAY ---
            if self.fade_counter > 0:
                cv2.putText(self.frame, self.overlay_text, (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, self.overlay_color, 4)
            # --- TOTAL SCORE DISPLAY ---
            score_text = f"Score: {self.makes}/{self.attempts}"
            cv2.putText(self.frame, score_text, (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

            self.frame_count += 1
            cv2.imshow("Frame", self.frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        self.cap.release()
        cv2.destroyAllWindows()

    # ------------------------ CLEAN / DETECT ------------------------
    def clean_motion(self):
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for b in self.ball_pos:
            cv2.circle(self.frame, b[0], 2, (255,0,255), 2)
            if len(self.hoop_pos) > 1:
                self.hoop_pos = clean_hoop_pos(self.hoop_pos)
                cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (0, 128, 0), 2)

    def shot_detection(self):

        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]
                    self.release_pos = self.ball_pos[-1][0]
            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]
            if self.frame_count % 20 == 0:
                if self.up and self.down and self.up_frame < self.down_frame:
                    self.attempts += 1
                    self.up = False
                    self.down = False
                    made = score(self.ball_pos, self.hoop_pos)
                    if made:
                        self.makes += 1
                        self.overlay_text = "Score"
                        self.overlay_color = (0, 255, 0)
                    else:
                        self.overlay_text = "Miss"
                        self.overlay_color = (0, 0, 255)
                    self.fade_counter = self.fade_frames
                    if hasattr(self, "release_pos"):
                        fh, fw = self.frame.shape[:2]
                        mh, mw = self.minimap_img.shape[:2]
                        mx = int(self.release_pos[0] / fw * mw)
                        my = int(self.release_pos[1] / fh * mh)
                        self.shot_history.append((mx, my, made))
        # Fade counter decrement
        if self.fade_counter > 0:
            self.fade_counter -= 1
    # ------------------------ MINIMAP ------------------------
    def draw_minimap(self, players=[]):
        minimap_copy = self.minimap_img.copy()
        for (x, y, made) in self.shot_history:
            color = (0, 255, 0) if made else (0, 0, 255)
            cv2.line(minimap_copy, (x - 10, y - 10), (x + 10, y + 10), color, 2)
            cv2.line(minimap_copy, (x - 10, y + 10), (x + 10, y - 10), color, 2)
        # Oyuncular
        for cx, cy in players:
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            proj_pt = cv2.perspectiveTransform(pt, self.H)[0][0]
            display_x = int(proj_pt[0])
            display_y = int(proj_pt[1])
            if self.use_flip:
                display_y = self.h_img - display_y
            inside = (0 <= display_x < minimap_copy.shape[1] and 0 <= display_y < minimap_copy.shape[0])
            color = (0, 0, 255) if inside else (0, 0, 128)
            cv2.circle(minimap_copy, (np.clip(display_x, 0, minimap_copy.shape[1] - 1),
                                      np.clip(display_y, 0, minimap_copy.shape[0] - 1)), 8, color, -1)
            if not inside:
                cv2.putText(minimap_copy, "OUT", (np.clip(display_x, 0, minimap_copy.shape[1] - 1) + 6,
                                                  np.clip(display_y, 0, minimap_copy.shape[0] - 1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Minimap AR", minimap_copy)

if __name__ == "__main__":
    ShotDetector()
