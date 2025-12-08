"""
3-Point Line Selection Tool
============================
hom.png üzerinde 3'lük çizgisini manuel olarak seçip JSON'a kaydeder.

Kullanım:
1. Script'i çalıştır
2. 3'lük çizgisi üzerindeki noktaları tıkla (saat yönünde veya tersi)
3. 'q' tuşuna basarak bitir
4. Koordinatlar three_point_line.json'a kaydedilir
"""

import cv2
import json
import os
import numpy as np

# Seçilen noktalar
points_3pt = []
temp_img = None
original_img = None

def mouse_callback(event, x, y, flags, param):
    global points_3pt, temp_img, original_img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        points_3pt.append([x, y])
        print(f"✓ Nokta eklendi: ({x}, {y}) - Toplam: {len(points_3pt)}")
        
        # Görselleştir
        temp_img = original_img.copy()
        
        # Noktaları çiz
        for i, pt in enumerate(points_3pt):
            cv2.circle(temp_img, tuple(pt), 4, (0, 255, 0), -1)
            cv2.putText(temp_img, str(i), (pt[0]+5, pt[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Eğer 2'den fazla nokta varsa poligon çiz
        if len(points_3pt) > 1:
            pts_array = np.array(points_3pt, np.int32)
            cv2.polylines(temp_img, [pts_array], False, (255, 0, 0), 2)
        
        cv2.imshow("3PT Line Selection", temp_img)

def main():
    global temp_img, original_img, points_3pt
    
    # hom.png'yi yükle
    base_dir = os.path.dirname(__file__)
    hom_path = os.path.normpath(os.path.join(base_dir, "..", "Homography", "images", "hom.png"))
    
    if not os.path.exists(hom_path):
        print(f"❌ Hata: {hom_path} bulunamadı!")
        return
    
    original_img = cv2.imread(hom_path)
    if original_img is None:
        print(f"❌ Hata: {hom_path} okunamadı!")
        return
    
    temp_img = original_img.copy()
    
    print("=" * 60)
    print("3-POINT LINE SELECTION TOOL")
    print("=" * 60)
    print("\n📍 Talimatlar:")
    print("  1. 3'lük çizgisi üzerindeki noktaları tıklayın")
    print("  2. Çizgiyi TAM OLARAK takip edin (yaklaşık 15-20 nokta yeterli)")
    print("  3. Sol taraftan başlayıp sağa doğru devam edin (veya tersi)")
    print("  4. Bitirdikten sonra 'q' tuşuna basın")
    print("  5. Son noktayı silmek için 'z' tuşuna basın")
    print("\n🎯 ÖNEMLİ: Potayı da içeren kapalı bir alan oluşturun!\n")
    
    cv2.namedWindow("3PT Line Selection")
    cv2.setMouseCallback("3PT Line Selection", mouse_callback)
    cv2.imshow("3PT Line Selection", temp_img)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # Quit
            if len(points_3pt) < 3:
                print("⚠️  En az 3 nokta seçmelisiniz!")
                continue
            break
        elif key == ord('z'):  # Undo
            if points_3pt:
                removed = points_3pt.pop()
                print(f"↶ Son nokta silindi: {removed} - Kalan: {len(points_3pt)}")
                
                # Yeniden çiz
                temp_img = original_img.copy()
                for i, pt in enumerate(points_3pt):
                    cv2.circle(temp_img, tuple(pt), 4, (0, 255, 0), -1)
                    cv2.putText(temp_img, str(i), (pt[0]+5, pt[1]-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                if len(points_3pt) > 1:
                    pts_array = np.array(points_3pt, np.int32)
                    cv2.polylines(temp_img, [pts_array], False, (255, 0, 0), 2)
                cv2.imshow("3PT Line Selection", temp_img)
        elif key == 27:  # ESC
            print("❌ İptal edildi.")
            cv2.destroyAllWindows()
            return
    
    cv2.destroyAllWindows()
    
    # JSON'a kaydet
    output_path = os.path.join(base_dir, "three_point_line.json")
    
    data = {
        "description": "3-point line coordinates on hom.png minimap",
        "note": "Points define the 3PT arc. Inside = 2PT, Outside = 3PT",
        "points": points_3pt
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Kaydedildi: {output_path}")
    print(f"📊 Toplam nokta sayısı: {len(points_3pt)}")
    
    # Önizleme göster
    preview = original_img.copy()
    pts_array = np.array(points_3pt, np.int32)
    
    # Kapalı poligon olarak çiz
    cv2.polylines(preview, [pts_array], True, (0, 255, 0), 2)
    
    # Alan doldur (yarı saydam)
    overlay = preview.copy()
    cv2.fillPoly(overlay, [pts_array], (0, 255, 0))
    cv2.addWeighted(overlay, 0.3, preview, 0.7, 0, preview)
    
    cv2.imshow("3PT Line - Preview", preview)
    print("\n✓ Önizleme görüntüleniyor. Kapatmak için bir tuşa basın...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n🎯 Artık shot_detector.py bu JSON'u kullanacak!")

if __name__ == "__main__":
    main()




