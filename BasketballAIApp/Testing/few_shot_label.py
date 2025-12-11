"""
OWL-ViT ile Few-Shot Object Detection
Örnek resimler vererek daha iyi tespit yapar
"""

import torch
from PIL import Image
import cv2
import os
from pathlib import Path
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from tqdm import tqdm

# ============== AYARLAR ==============
IMAGES_DIR = r"D:\repos\Basketball_App\BasketballAIApp\Datasets\frames_output\hepsi"
LABELS_DIR = r"D:\repos\Basketball_App\BasketballAIApp\Datasets\frames_output\labels_fewshot"
EXAMPLES_DIR = r"D:\repos\Basketball_App\BasketballAIApp\Testing\examples"  # Örnek resimler klasörü
CONFIDENCE_THRESHOLD = 0.1  # Few-shot için düşük threshold

# Sınıflar
CLASS_NAMES = ["basketball", "rim", "player"]
# =====================================

def setup_model():
    """Model ve processor'ı yükle"""
    print("OWL-ViT v2 modeli yükleniyor...")
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}")
    
    return model, processor, device

def load_example_images(examples_dir):
    """
    Örnek resimleri yükle
    Klasör yapısı:
    examples/
        basketball/
            example1.jpg
            example2.jpg
        rim/
            example1.jpg
        player/
            example1.jpg
            example2.jpg
    """
    example_images = {}
    
    for class_name in CLASS_NAMES:
        class_dir = Path(examples_dir) / class_name
        if class_dir.exists():
            images = []
            for img_path in class_dir.glob("*.*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
            if images:
                example_images[class_name] = images
                print(f"  {class_name}: {len(images)} örnek yüklendi")
    
    return example_images

def detect_with_examples(image_path, model, processor, device, example_images):
    """Few-shot detection with example images"""
    target_image = Image.open(image_path).convert("RGB")
    w, h = target_image.size
    
    detections = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        if class_name not in example_images:
            continue
        
        query_images = example_images[class_name]
        
        # Process inputs
        inputs = processor(
            images=target_image,
            query_images=query_images,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model.image_guided_detection(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([[h, w]], device=device)
        results = processor.post_process_image_guided_detection(
            outputs=outputs,
            threshold=CONFIDENCE_THRESHOLD,
            nms_threshold=0.3,
            target_sizes=target_sizes
        )[0]
        
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            
            # YOLO formatına çevir
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            box_width = (x2 - x1) / w
            box_height = (y2 - y1) / h
            
            detections.append({
                "class_id": class_idx,
                "class_name": class_name,
                "x_center": x_center,
                "y_center": y_center,
                "width": box_width,
                "height": box_height,
                "confidence": float(score)
            })
    
    return detections

def detect_with_text(image_path, model, processor, device):
    """Text-based detection (fallback if no examples)"""
    target_image = Image.open(image_path).convert("RGB")
    w, h = target_image.size
    
    texts = [["basketball ball", "basketball hoop rim ring", "person basketball player"]]
    
    inputs = processor(
        images=target_image,
        text=texts,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([[h, w]], device=device)
    results = processor.post_process_object_detection(
        outputs=outputs,
        threshold=CONFIDENCE_THRESHOLD,
        target_sizes=target_sizes
    )[0]
    
    detections = []
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    
    for box, score, label_idx in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        box_width = (x2 - x1) / w
        box_height = (y2 - y1) / h
        
        detections.append({
            "class_id": int(label_idx),
            "class_name": CLASS_NAMES[label_idx] if label_idx < len(CLASS_NAMES) else f"class_{label_idx}",
            "x_center": x_center,
            "y_center": y_center,
            "width": box_width,
            "height": box_height,
            "confidence": float(score)
        })
    
    return detections

def save_yolo_labels(detections, label_path):
    """YOLO formatında kaydet"""
    with open(label_path, "w") as f:
        for det in detections:
            line = f"{det['class_id']} {det['x_center']:.6f} {det['y_center']:.6f} {det['width']:.6f} {det['height']:.6f}\n"
            f.write(line)

def main():
    os.makedirs(LABELS_DIR, exist_ok=True)
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    
    # Örnek klasörlerini oluştur
    for class_name in CLASS_NAMES:
        os.makedirs(Path(EXAMPLES_DIR) / class_name, exist_ok=True)
    
    # Örnek resimleri yükle
    print("\nÖrnek resimler yükleniyor...")
    example_images = load_example_images(EXAMPLES_DIR)
    
    if not example_images:
        print("\n⚠️  UYARI: Örnek resim bulunamadı!")
        print(f"Lütfen şu klasöre örnek resimler ekleyin: {EXAMPLES_DIR}")
        print("\nKlasör yapısı:")
        print("  examples/")
        print("    basketball/")
        print("      basketbol_topu_1.jpg")
        print("      basketbol_topu_2.jpg")
        print("    rim/")
        print("      cember_1.jpg")
        print("    player/")
        print("      oyuncu_1.jpg")
        print("\nText-based detection kullanılacak...")
        use_examples = False
    else:
        use_examples = True
        print(f"\n✓ {len(example_images)} sınıf için örnek yüklendi")
    
    # Model yükle
    model, processor, device = setup_model()
    
    # Resimleri bul
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(IMAGES_DIR).glob(f"*{ext}"))
        image_files.extend(Path(IMAGES_DIR).glob(f"*{ext.upper()}"))
    
    image_files = list(set(image_files))
    print(f"\nToplam {len(image_files)} resim bulundu.")
    
    stats = {name: 0 for name in CLASS_NAMES}
    stats["total_images"] = 0
    
    print("\nEtiketleme başlıyor...\n")
    
    for image_path in tqdm(image_files, desc="Etiketleniyor"):
        try:
            if use_examples:
                detections = detect_with_examples(image_path, model, processor, device, example_images)
            else:
                detections = detect_with_text(image_path, model, processor, device)
            
            label_path = Path(LABELS_DIR) / (image_path.stem + ".txt")
            save_yolo_labels(detections, label_path)
            
            stats["total_images"] += 1
            for det in detections:
                if det["class_name"] in stats:
                    stats[det["class_name"]] += 1
                    
        except Exception as e:
            print(f"\nHata ({image_path.name}): {e}")
    
    print("\n" + "=" * 50)
    print("ETİKETLEME TAMAMLANDI!")
    print("=" * 50)
    print(f"Toplam resim: {stats['total_images']}")
    for name in CLASS_NAMES:
        print(f"Tespit edilen {name}: {stats[name]}")
    print(f"\nLabel dosyaları: {LABELS_DIR}")
    
    # classes.txt
    classes_file = Path(LABELS_DIR).parent / "classes.txt"
    with open(classes_file, "w") as f:
        for name in CLASS_NAMES:
            f.write(f"{name}\n")

if __name__ == "__main__":
    main()


