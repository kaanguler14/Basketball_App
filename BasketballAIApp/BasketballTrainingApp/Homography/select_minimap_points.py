import cv2
import json
import tkinter as tk
from tkinter import ttk

# Seçilecek noktalar
court_labels = [
    "top_left", "top_right", "bottom_right", "bottom_left",
    "free_throw_left", "free_throw_right",
    "center_circle", "paint_left", "paint_right"
]

clicks = {}
current_label = None

# Tkinter ile label seçtir
def ask_label():
    root = tk.Tk()
    root.title("Minimap Nokta Seçimi")
    var = tk.StringVar(value=court_labels[0])

    tk.Label(root, text="Bu nokta hangisi?").pack(padx=10, pady=5)
    combo = ttk.Combobox(root, values=court_labels, textvariable=var, state="readonly")
    combo.pack(padx=10, pady=5)

    def submit():
        root.quit()

    tk.Button(root, text="Seç", command=submit).pack(pady=5)

    root.mainloop()
    choice = var.get()
    root.destroy()
    return choice

# Mouse callback
def on_mouse(event, x, y, flags, param):
    global current_label
    if event == cv2.EVENT_LBUTTONDOWN:
        # Kullanıcıdan hangi nokta olduğunu sor
        label = ask_label()
        clicks[label] = (x, y)
        print(f"{label} seçildi: ({x},{y})")

def main():
    img = cv2.imread(r"D:\repos\Basketball_App\BasketballAIApp\BasketballTrainingApp\Homography\images\hom.png")
    if img is None:
        print("hom.png bulunamadı!")
        return

    cv2.namedWindow("Minimap")
    cv2.setMouseCallback("Minimap", on_mouse)

    print("Noktalara tıkla, her seferinde label seç. ESC ile bitir.")

    while True:
        tmp = img.copy()
        for label, pt in clicks.items():
            cv2.circle(tmp, pt, 6, (0, 0, 255), -1)
            cv2.putText(tmp, label, (pt[0]+6, pt[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Minimap", tmp)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break

    cv2.destroyAllWindows()

    # JSON olarak kaydet
    with open("minimap_points.json", "w") as f:
        json.dump(clicks, f, indent=2)

    print("Minimap noktaları kaydedildi → minimap_points.json")

if __name__ == "__main__":
    main()
