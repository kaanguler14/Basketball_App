"""
YOLO etiketlerini resimler üzerine çizip kaydeder
"""

import cv2
import os
from pathlib import Path

# ============== AYARLAR ==============
IMAGES_DIR = r"D:\repos\Basketball_App\BasketballAIApp\Datasets\frames_output\hepsi"
LABELS_DIR = r"D:\repos\Basketball_App\BasketballAIApp\Datasets\frames_output\labels"
OUTPUT_DIR = r"D:\repos\Basketball_App\BasketballAIApp\Datasets\frames_output\visualized"

# Sınıf isimleri ve renkleri
CLASSES = {
    0: ("basketball", (0, 165, 255)),   # Turuncu
    1: ("rim", (0, 255, 0)),             # Yeşil
    2: ("player", (255, 0, 0)),          # Mavi
}
# =====================================

def read_yolo_labels(label_path):
    """YOLO label dosyasını oku"""
    labels = []
    if not os.path.exists(label_path):
        return labels
    
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                labels.append({
                    "class_id": class_id,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height
                })
    return labels

def draw_labels(image, labels):
    """Resim üzerine bounding box çiz"""
    h, w = image.shape[:2]
    
    for label in labels:
        class_id = label["class_id"]
        x_center = label["x_center"] * w
        y_center = label["y_center"] * h
        box_w = label["width"] * w
        box_h = label["height"] * h
        
        # Köşe koordinatları
        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)
        
        # Sınıf bilgisi
        class_name, color = CLASSES.get(class_id, (f"class_{class_id}", (255, 255, 255)))
        
        # Bounding box çiz
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Label arka planı ve yazısı
        label_text = class_name
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), color, -1)
        cv2.putText(image, label_text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image

def main():
    # Output klasörünü oluştur
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Tüm resimleri bul
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(IMAGES_DIR).glob(f"*{ext}"))
        image_files.extend(Path(IMAGES_DIR).glob(f"*{ext.upper()}"))
    
    image_files = list(set(image_files))
    print(f"Toplam {len(image_files)} resim bulundu.")
    
    processed = 0
    skipped = 0
    
    for image_path in image_files:
        # Label dosya yolu
        label_path = Path(LABELS_DIR) / (image_path.stem + ".txt")
        
        # Resmi oku
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Resim okunamadı: {image_path.name}")
            skipped += 1
            continue
        
        # Labelleri oku
        labels = read_yolo_labels(label_path)
        
        if len(labels) == 0:
            skipped += 1
            continue
        
        # Çiz
        image = draw_labels(image, labels)
        
        # Kaydet
        output_path = Path(OUTPUT_DIR) / image_path.name
        cv2.imwrite(str(output_path), image)
        processed += 1
        
        if processed % 50 == 0:
            print(f"İşlenen: {processed}/{len(image_files)}")
    
    print("\n" + "=" * 50)
    print("GÖRSELLEŞTIRME TAMAMLANDI!")
    print("=" * 50)
    print(f"İşlenen resim: {processed}")
    print(f"Atlanan resim: {skipped}")
    print(f"\nKaydedilen klasör: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

