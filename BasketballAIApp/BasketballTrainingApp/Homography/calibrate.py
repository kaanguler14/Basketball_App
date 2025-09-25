import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

# --- Kullanıcı seçimi için GUI ---
court_labels = [
    "top_left", "top_right", "bottom_right", "bottom_left",
    "free_throw_left", "free_throw_right", "center_circle",
    "paint_left", "paint_right"
]

clicks = []
labels = []

def ask_label():
    root = tk.Tk()
    root.title("Court Point Seçimi")
    var = tk.StringVar(value=court_labels[0])

    label = tk.Label(root, text="Bu nokta neresi?")
    label.pack(padx=10, pady=5)

    combo = ttk.Combobox(root, values=court_labels, textvariable=var, state="readonly")
    combo.pack(padx=10, pady=5)

    def submit():
        root.quit()

    btn = tk.Button(root, text="Seç", command=submit)
    btn.pack(pady=5)

    root.mainloop()
    choice = var.get()
    root.destroy()
    return choice

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        label = ask_label()
        clicks.append((x, y))
        labels.append(label)
        print(f"Kaydedildi: {label} → ({x},{y})")

# --- Ana pipeline ---
def main():
    cap = cv2.VideoCapture(r"D:\repos\Basketball_App\BasketballAIApp\clips\training7.mp4")
    ret, frame = cap.read()
    if not ret:
        print("Video okunamadı")
        return

    # Frame'i küçült
    scale = 0.5
    frame = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))

    cv2.namedWindow("Court Calibration")
    cv2.setMouseCallback("Court Calibration", on_mouse)

    print("Sahadan bazı noktaları seçin (ESC çıkış)")

    while True:
        tmp = frame.copy()
        for (pt, label) in zip(clicks, labels):
            cv2.circle(tmp, pt, 6, (0, 255, 0), -1)
            cv2.putText(tmp, label, (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Court Calibration", tmp)

        k = cv2.waitKey(1) & 0xFF
        if k == 27 and len(clicks) >= 4:  # ESC ve en az 4 nokta seçilmişse
            break

    cv2.destroyWindow("Court Calibration")

    # --- Sahadaki şablon noktaları (metre cinsinden) ---
    court_template = {
        "top_left": (0, 0),
        "top_right": (28, 0),
        "bottom_right": (28, 15),
        "bottom_left": (0, 15),
        "free_throw_left": (5.8, 7.5),
        "free_throw_right": (22.2, 7.5),
        "center_circle": (14, 7.5),
        "paint_left": (4.225, 7.5),
        "paint_right": (23.775, 7.5)
    }

    # Video noktaları + saha noktaları eşleştirme
    video_points = np.array(clicks, dtype=np.float32)
    template_points = np.array([court_template[l] for l in labels], dtype=np.float32)

    # Homography matrisi
    H, _ = cv2.findHomography(video_points, template_points)
    if H is None:
        print("Homography hesaplanamadı")
        return

    # --- AR DEMO: Oyuncu noktası (örnek piksel koordinatı) ---
    example_px = np.array([[[frame.shape[1]//2, frame.shape[0]//2]]], dtype=np.float32)  # ortada bir nokta
    example_field = cv2.perspectiveTransform(example_px, H)[0][0]

    print("Oyuncu saha koordinatı:", example_field)

    # --- Minimap çiz ---
    minimap = np.ones((300, 560, 3), dtype=np.uint8) * 255  # beyaz arka plan
    scale_x = 560 / 28.0
    scale_y = 300 / 15.0

    # Oyuncu noktası saha koordinatından minimap'e
    px = int(example_field[0] * scale_x)
    py = int(example_field[1] * scale_y)
    cv2.circle(minimap, (px, py), 8, (0, 0, 255), -1)

    cv2.imshow("Minimap AR", minimap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()
