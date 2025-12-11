import torch
from PIL import Image
import cv2
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

IMAGE_PATH = r"D:\repos\Basketball_App\BasketballAIApp\Datasets\etiketli\images\training2_sec0003.jpg"

print("Grounding DINO modeli yükleniyor...")
model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

# GPU varsa kullan
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Device: {device}")

# Resmi yükle
image = Image.open(IMAGE_PATH)
img_cv = cv2.imread(IMAGE_PATH)
h, w = img_cv.shape[:2]

# Tespit edilecek nesneler (metin ile belirt)
text = "basketball. basketball rim. person."

print(f"Aranan nesneler: {text}")
print("Tespit yapılıyor...")

# Inference
inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

# Post-process
results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    target_sizes=[(h, w)]
)[0]

# Threshold uygula
mask = results["scores"] > 0.3
results = {
    "boxes": results["boxes"][mask],
    "scores": results["scores"][mask],
    "labels": [label for label, m in zip(results["labels"], mask) if m]
}

# Renk paleti
COLORS = {
    "basketball": (0, 165, 255),   # Turuncu
    "basketball rim": (0, 255, 0),  # Yeşil
    "rim": (0, 255, 0),             # Yeşil
    "person": (255, 0, 0),          # Mavi
}

print("\nTESPİT EDİLEN NESNELER:")
print("-" * 50)

boxes = results["boxes"].cpu().numpy()
scores = results["scores"].cpu().numpy()
labels = results["labels"]

for box, score, label in zip(boxes, scores, labels):
    x1, y1, x2, y2 = map(int, box)
    
    # En yakın rengi bul
    color = (255, 255, 255)
    for key in COLORS:
        if key in label.lower():
            color = COLORS[key]
            break
    
    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
    label_text = f"{label} {score:.2f}"
    
    # Label arka planı
    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img_cv, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
    cv2.putText(img_cv, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    print(f"  - {label}: conf={score:.2f}, box=[{x1}, {y1}, {x2}, {y2}]")

print("-" * 50)

# Kaydet ve göster
output_path = r"D:\repos\Basketball_App\BasketballAIApp\Testing\detection_result.jpg"
cv2.imwrite(output_path, img_cv)
print(f"\nSonuç kaydedildi: {output_path}")

cv2.imshow("Grounding DINO Detection", img_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
