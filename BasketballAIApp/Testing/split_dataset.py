"""
Dataset'i train/val/test olarak böler
YOLO eğitimi için hazırlar
"""

import os
import shutil
import random
from pathlib import Path

# ============== AYARLAR ==============
IMAGES_DIR = r"D:\repos\Basketball_App\BasketballAIApp\Datasets\frames_output\hepsi"
LABELS_DIR = r"D:\repos\Basketball_App\BasketballAIApp\Datasets\frames_output\labels"
OUTPUT_DIR = r"D:\repos\Basketball_App\BasketballAIApp\Datasets\yolo_dataset"

# Bölme oranları (toplam 1.0 olmalı)
TRAIN_RATIO = 0.7   # %70 train
VAL_RATIO = 0.2     # %20 validation
TEST_RATIO = 0.1    # %10 test

# Sınıf isimleri
CLASS_NAMES = ["basketball", "rim", "player"]

# Random seed (tekrarlanabilirlik için)
RANDOM_SEED = 42
# =====================================

def create_yaml_config(output_dir, class_names):
    """YOLO için data.yaml oluştur"""
    yaml_content = f"""# YOLO Dataset Configuration
path: {output_dir}
train: images/train
val: images/val
test: images/test

# Classes
names:
"""
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"
    
    yaml_path = Path(output_dir) / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    return yaml_path

def split_dataset():
    random.seed(RANDOM_SEED)
    
    # Tüm resimleri bul
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(Path(IMAGES_DIR).glob(ext))
    
    # Sadece etiketi olan resimleri al
    valid_files = []
    for img_path in image_files:
        label_path = Path(LABELS_DIR) / (img_path.stem + ".txt")
        if label_path.exists():
            # Boş olmayan etiketleri kontrol et
            with open(label_path) as f:
                if f.read().strip():
                    valid_files.append(img_path)
    
    print(f"Toplam etiketli resim: {len(valid_files)}")
    
    # Karıştır
    random.shuffle(valid_files)
    
    # Böl
    total = len(valid_files)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    splits = {
        "train": valid_files[:train_end],
        "val": valid_files[train_end:val_end],
        "test": valid_files[val_end:]
    }
    
    print(f"\nBölme sonuçları:")
    print(f"  Train: {len(splits['train'])} resim")
    print(f"  Val:   {len(splits['val'])} resim")
    print(f"  Test:  {len(splits['test'])} resim")
    
    # Klasör yapısını oluştur
    for split_name in ["train", "val", "test"]:
        os.makedirs(Path(OUTPUT_DIR) / "images" / split_name, exist_ok=True)
        os.makedirs(Path(OUTPUT_DIR) / "labels" / split_name, exist_ok=True)
    
    # Dosyaları kopyala
    print("\nDosyalar kopyalanıyor...")
    
    for split_name, files in splits.items():
        for img_path in files:
            # Resmi kopyala
            dest_img = Path(OUTPUT_DIR) / "images" / split_name / img_path.name
            shutil.copy2(img_path, dest_img)
            
            # Etiketi kopyala
            label_path = Path(LABELS_DIR) / (img_path.stem + ".txt")
            dest_label = Path(OUTPUT_DIR) / "labels" / split_name / (img_path.stem + ".txt")
            shutil.copy2(label_path, dest_label)
    
    # YAML config oluştur
    yaml_path = create_yaml_config(OUTPUT_DIR, CLASS_NAMES)
    
    # classes.txt oluştur
    classes_path = Path(OUTPUT_DIR) / "classes.txt"
    with open(classes_path, "w") as f:
        for name in CLASS_NAMES:
            f.write(f"{name}\n")
    
    print("\n" + "=" * 50)
    print("DATASET HAZIR!")
    print("=" * 50)
    print(f"\n📁 Çıktı klasörü: {OUTPUT_DIR}")
    print(f"📄 YOLO config: {yaml_path}")
    print(f"\n📊 İstatistikler:")
    print(f"   Train: {len(splits['train'])} resim ({TRAIN_RATIO*100:.0f}%)")
    print(f"   Val:   {len(splits['val'])} resim ({VAL_RATIO*100:.0f}%)")
    print(f"   Test:  {len(splits['test'])} resim ({TEST_RATIO*100:.0f}%)")
    
    print(f"\n🚀 YOLO eğitimi için kullanım:")
    print(f"   yolo detect train data={yaml_path} model=yolov8n.pt epochs=100")
    
    return OUTPUT_DIR

if __name__ == "__main__":
    split_dataset()

