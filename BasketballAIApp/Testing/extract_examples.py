"""
YOLO Pose etiketlerinden örnek resimleri otomatik kırpar
Few-shot learning için örnek klasörlerini doldurur
"""

import cv2
import os
from pathlib import Path
from tqdm import tqdm
import random

# ============== AYARLAR ==============
# Pose etiketli resimler (player için)
POSE_IMAGES_DIR = r"D:\repos\Basketball_App\BasketballAIApp\Datasets\etiketli\train\images"
POSE_LABELS_DIR = r"D:\repos\Basketball_App\BasketballAIApp\Datasets\etiketli\train\labels"

# Grounding DINO etiketli resimler (basketball ve rim için)
DINO_IMAGES_DIR = r"D:\repos\Basketball_App\BasketballAIApp\Datasets\frames_output\hepsi"
DINO_LABELS_DIR = r"D:\repos\Basketball_App\BasketballAIApp\Datasets\frames_output\labels"

# Örnek resimlerin kaydedileceği klasör
EXAMPLES_DIR = r"D:\repos\Basketball_App\BasketballAIApp\Testing\examples"

# Sınıf isimleri
CLASS_NAMES = ["basketball", "rim", "player"]

# Her sınıf için maksimum örnek sayısı
MAX_EXAMPLES_PER_CLASS = 10

# Kırpma padding (piksel)
PADDING = 10
# =====================================

def read_pose_labels(label_path, img_width, img_height):
    """YOLO Pose label dosyasını oku (sadece bbox kısmı)"""
    labels = []
    
    if not os.path.exists(label_path):
        return labels
    
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(float(parts[0]))
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                labels.append({
                    "class_id": class_id,
                    "class_name": "player",  # Pose etiketleri player için
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })
    
    return labels

def read_yolo_labels(label_path, img_width, img_height):
    """Standard YOLO label dosyasını oku"""
    labels = []
    
    if not os.path.exists(label_path):
        return labels
    
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(float(parts[0]))
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                # Class mapping: 0=basketball, 1=rim, 2=player
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
                
                labels.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })
    
    return labels

def crop_and_save(image, label, output_dir, index):
    """Nesneyi kırp ve kaydet"""
    h, w = image.shape[:2]
    
    x1 = max(0, label["x1"] - PADDING)
    y1 = max(0, label["y1"] - PADDING)
    x2 = min(w, label["x2"] + PADDING)
    y2 = min(h, label["y2"] + PADDING)
    
    cropped = image[y1:y2, x1:x2]
    
    if cropped.size == 0 or cropped.shape[0] < 20 or cropped.shape[1] < 20:
        return False
    
    class_dir = Path(output_dir) / label["class_name"]
    os.makedirs(class_dir, exist_ok=True)
    
    output_path = class_dir / f"example_{index}.jpg"
    cv2.imwrite(str(output_path), cropped)
    
    return True

def main():
    # Örnek klasörlerini oluştur ve temizle
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    for class_name in CLASS_NAMES:
        class_dir = Path(EXAMPLES_DIR) / class_name
        os.makedirs(class_dir, exist_ok=True)
        for f in class_dir.glob("*.jpg"):
            f.unlink()
    
    all_crops = {name: [] for name in CLASS_NAMES}
    
    # 1. POSE etiketlerinden PLAYER örnekleri topla
    print("=" * 50)
    print("1. POSE etiketlerinden PLAYER örnekleri toplanıyor...")
    print("=" * 50)
    
    if os.path.exists(POSE_IMAGES_DIR):
        image_files = list(Path(POSE_IMAGES_DIR).glob("*.jpg"))
        print(f"Toplam {len(image_files)} pose etiketli resim bulundu.")
        
        for image_path in tqdm(image_files, desc="Pose taranıyor"):
            label_path = Path(POSE_LABELS_DIR) / (image_path.stem + ".txt")
            
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            h, w = image.shape[:2]
            labels = read_pose_labels(label_path, w, h)
            
            for label in labels:
                all_crops["player"].append({
                    "image_path": image_path,
                    "label": label
                })
    else:
        print(f"⚠️  Pose klasörü bulunamadı: {POSE_IMAGES_DIR}")
    
    # 2. DINO etiketlerinden BASKETBALL ve RIM örnekleri topla
    print("\n" + "=" * 50)
    print("2. Grounding DINO etiketlerinden BASKETBALL ve RIM örnekleri toplanıyor...")
    print("=" * 50)
    
    if os.path.exists(DINO_LABELS_DIR):
        label_files = list(Path(DINO_LABELS_DIR).glob("*.txt"))
        print(f"Toplam {len(label_files)} DINO etiket dosyası bulundu.")
        
        for label_path in tqdm(label_files, desc="DINO taranıyor"):
            image_path = Path(DINO_IMAGES_DIR) / (label_path.stem + ".jpg")
            
            if not image_path.exists():
                # PNG olarak dene
                image_path = Path(DINO_IMAGES_DIR) / (label_path.stem + ".png")
            
            if not image_path.exists():
                continue
            
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            h, w = image.shape[:2]
            labels = read_yolo_labels(label_path, w, h)
            
            for label in labels:
                class_name = label["class_name"]
                if class_name in ["basketball", "rim"]:
                    all_crops[class_name].append({
                        "image_path": image_path,
                        "label": label
                    })
    else:
        print(f"⚠️  DINO labels klasörü bulunamadı: {DINO_LABELS_DIR}")
        print("Önce 'python auto_label.py' çalıştırın!")
    
    # 3. Rastgele örnekler seç ve kaydet
    print("\n" + "=" * 50)
    print("3. Örnekler seçiliyor ve kaydediliyor...")
    print("=" * 50)
    
    class_counts = {}
    for class_name in CLASS_NAMES:
        crops = all_crops[class_name]
        
        if len(crops) > MAX_EXAMPLES_PER_CLASS:
            crops = random.sample(crops, MAX_EXAMPLES_PER_CLASS)
        
        count = 0
        for i, crop_info in enumerate(crops):
            image = cv2.imread(str(crop_info["image_path"]))
            if image is not None:
                if crop_and_save(image, crop_info["label"], EXAMPLES_DIR, i + 1):
                    count += 1
        
        class_counts[class_name] = count
        print(f"  {class_name}: {count} örnek kaydedildi")
    
    # Sonuç
    print("\n" + "=" * 50)
    print("ÖRNEKLER HAZIR!")
    print("=" * 50)
    print(f"Kaydedilen klasör: {EXAMPLES_DIR}")
    
    missing = [name for name in CLASS_NAMES if class_counts.get(name, 0) == 0]
    if missing:
        print(f"\n⚠️  UYARI: Şu sınıflar için örnek bulunamadı: {missing}")
        if "basketball" in missing or "rim" in missing:
            print("\n💡 Basketball ve rim için önce Grounding DINO çalıştırın:")
            print("   python auto_label.py")
    
    print("\n📌 Şimdi few_shot_label.py çalıştırabilirsiniz!")

if __name__ == "__main__":
    main()
