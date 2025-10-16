import os
import shutil
import random

# --- Config ---
images_dir = r"D:\repos\Basketball_App\BasketballAIApp\Datasets\etiketli\images"
labels_dir = r"D:\repos\Basketball_App\BasketballAIApp\Datasets\etiketli\yolola"
output_dir = r"D:\repos\Basketball_App\BasketballAIApp\Datasets\etiketli"

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# --- Dosya listesi ---
all_images = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg",".png"))]
random.shuffle(all_images)

num_total = len(all_images)
num_train = int(num_total * train_ratio)
num_val = int(num_total * val_ratio)

train_files = all_images[:num_train]
val_files = all_images[num_train:num_train+num_val]
test_files = all_images[num_train+num_val:]

# --- Fonksiyon: dosyaları kopyala ---
def copy_files(file_list, subset_name):
    img_out = os.path.join(output_dir, subset_name, "images")
    lbl_out = os.path.join(output_dir, subset_name, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for img_file in file_list:
        base_name = os.path.splitext(img_file)[0]
        lbl_file = base_name + ".txt"

        # Kopyala
        shutil.copy(os.path.join(images_dir, img_file), os.path.join(img_out, img_file))
        lbl_path = os.path.join(labels_dir, lbl_file)
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(lbl_out, lbl_file))
        else:
            print(f"[WARNING] Label yok: {lbl_file}")

# --- Kopyala ---
copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print("✅ Dataset train/val/test olarak ayrıldı.")
