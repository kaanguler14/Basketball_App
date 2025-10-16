import os
import json
import cv2
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk
court_labels = [
    "top_left", "top_right","top", "bottom_right", "bottom_left",
    "free_throw_left", "free_throw_right",
    "center_circle", "paint_left", "paint_right"
]
minimap_img = cv2.imread(
    r"D:\repos\Basketball_App\BasketballAIApp\BasketballTrainingApp\images\hom.png")

vp_json = "video_points.json"
mp_json = "minimap_points.json"

def load_or_select(video_frame):
    video_points_dict, minimap_points_dict = {}, {}

    if os.path.exists(vp_json) and os.path.exists(mp_json):
        if ask_reload():
            video_points_dict = select_points(video_frame, "Video")
            with open(vp_json, "w") as f:
                json.dump(video_points_dict, f, indent=2)
            print("Video noktaları güncellendi.")
        else:
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

    video_points_dict = {k: (int(v[0]), int(v[1])) for k, v in video_points_dict.items()}
    minimap_points_dict = {k: (int(v[0]), int(v[1])) for k, v in minimap_points_dict.items()}
    return video_points_dict, minimap_points_dict


def ask_reload():
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesno("JSON bulundu", "Önceden kaydedilmiş noktalar bulundu.\nYeniden seçmek ister misin?")
    root.destroy()
    return result


def select_points(image, title):
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
            cv2.putText(tmp, label, (pt[0] + 6, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow(title, tmp)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyWindow(title)
    return points_dict
