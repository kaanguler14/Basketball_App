"""
Grounding DINO ile otomatik YOLO etiketleme
Tüm resimleri tarar ve YOLO formatında .txt dosyaları oluşturur
"""

import torch
from PIL import Image
import cv2
import os
from pathlib import Path
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from tqdm import tqdm

# ============== AYARLAR ==============
IMAGES_DIR = r"D:\repos\Basketball_App\BasketballAIApp\Datasets\frames_output\hepsi"
LABELS_DIR = r"D:\repos\Basketball_App\BasketballAIApp\Datasets\frames_output\labels"
CONFIDENCE_THRESHOLD = 0.3

# Tespit edilecek sınıflar ve YOLO class ID'leri
CLASSES = {
    "basketball": 0,
    "basketball rim": 1,
    "rim": 1,  # Alternatif isim
    "player": 2,
    "person": 2,  # Alternatif isim
}

# Grounding DINO prompt (sadece 3 sınıf)
DETECTION_PROMPT = "orange basketball ball. orange basketball hoop ring. person playing basketball."
# =====================================

def setup_model():
    """Model ve processor'ı yükle"""
    print("Grounding DINO modeli yükleniyor...")
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}")
    
    return model, processor, device

def detect_objects(image_path, model, processor, device):
    """Tek bir resimde nesne tespiti yap"""
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    
    inputs = processor(images=image, text=DETECTION_PROMPT, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        target_sizes=[(h, w)]
    )[0]
    
    # Threshold uygula
    mask = results["scores"] > CONFIDENCE_THRESHOLD
    
    detections = []
    boxes = results["boxes"][mask].cpu().numpy()
    scores = results["scores"][mask].cpu().numpy()
    labels = [label for label, m in zip(results["labels"], mask) if m]
    
    for box, score, label in zip(boxes, scores, labels):
        # Label'ı class ID'ye çevir (önce spesifik olanları kontrol et)
        label_lower = label.lower().strip()
        
        if "ring" in label_lower or "hoop" in label_lower or "rim" in label_lower:
            class_id = 1  # rim (çember)
        elif "player" in label_lower or "person" in label_lower or "playing" in label_lower:
            class_id = 2  # player
        elif "ball" in label_lower or "basketball" in label_lower:
            class_id = 0  # basketball (sadece top)
        else:
            continue
        
        # YOLO formatına çevir: x_center, y_center, width, height (normalized)
        x1, y1, x2, y2 = box
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        box_width = (x2 - x1) / w
        box_height = (y2 - y1) / h
        
        detections.append({
            "class_id": class_id,
            "x_center": x_center,
            "y_center": y_center,
            "width": box_width,
            "height": box_height,
            "confidence": score,
            "label": label
        })
    
    return detections

def save_yolo_labels(detections, label_path):
    """YOLO formatında label dosyası kaydet"""
    with open(label_path, "w") as f:
        for det in detections:
            line = f"{det['class_id']} {det['x_center']:.6f} {det['y_center']:.6f} {det['width']:.6f} {det['height']:.6f}\n"
            f.write(line)

def main():
    # Labels klasörünü oluştur
    os.makedirs(LABELS_DIR, exist_ok=True)
    
    # Model yükle
    model, processor, device = setup_model()
    
    # Tüm resimleri bul
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(IMAGES_DIR).glob(f"*{ext}"))
        image_files.extend(Path(IMAGES_DIR).glob(f"*{ext.upper()}"))
    
    image_files = list(set(image_files))  # Duplikaları kaldır
    print(f"\nToplam {len(image_files)} resim bulundu.")
    
    # İstatistikler
    stats = {"basketball": 0, "rim": 0, "player": 0, "total_images": 0}
    
    print("\nEtiketleme başlıyor...\n")
    
    for image_path in tqdm(image_files, desc="Etiketleniyor"):
        try:
            detections = detect_objects(image_path, model, processor, device)
            
            # Label dosya yolu
            label_filename = image_path.stem + ".txt"
            label_path = Path(LABELS_DIR) / label_filename
            
            # Kaydet
            save_yolo_labels(detections, label_path)
            
            # İstatistik güncelle
            stats["total_images"] += 1
            for det in detections:
                if det["class_id"] == 0:
                    stats["basketball"] += 1
                elif det["class_id"] == 1:
                    stats["rim"] += 1
                elif det["class_id"] == 2:
                    stats["player"] += 1
                    
        except Exception as e:
            print(f"\nHata ({image_path.name}): {e}")
    
    # Sonuç özeti
    print("\n" + "=" * 50)
    print("ETİKETLEME TAMAMLANDI!")
    print("=" * 50)
    print(f"Toplam resim: {stats['total_images']}")
    print(f"Tespit edilen basketbol: {stats['basketball']}")
    print(f"Tespit edilen çember (rim): {stats['rim']}")
    print(f"Tespit edilen oyuncu (player): {stats['player']}")
    print(f"\nLabel dosyaları: {LABELS_DIR}")
    
    # classes.txt oluştur
    classes_file = Path(LABELS_DIR).parent / "classes.txt"
    with open(classes_file, "w") as f:
        f.write("basketball\n")
        f.write("rim\n")
        f.write("player\n")
    print(f"Sınıf dosyası: {classes_file}")

if __name__ == "__main__":
    main()

