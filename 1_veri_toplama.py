import cv2
import os

# ==========================================
# AYARLAR
# ==========================================
# Az önce kaydettiğin lokal videonun adı
VIDEO_PATH = r"D:\staj kod\task3\gunduz_video.mp4" 

# Fotoğrafların kaydedileceği klasörü oluştur
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ HATA: '{VIDEO_PATH}' dosyası bulunamadı! Lütfen videoyu task3 klasörüne koyduğundan emin ol.")
    exit()

# ==========================================
# GÖRÜNTÜLEME VE KAYDETME
# ==========================================
print("\n✅ VİDEO BAŞLADI!")
print("KULLANIM KILAVUZU:")
print("📸 Fotoğraf kaydetmek için klavyeden 's' (save) tuşuna bas.")
print("🚪 Çıkmak için 'q' (quit) tuşuna bas.\n")

# Klasördeki mevcut resim sayısını bul ve sayacı oradan başlat (Üstüne yazmayı engeller)
mevcut_resimler = [f for f in os.listdir(DATASET_DIR) if f.endswith('.jpg')]
count = len(mevcut_resimler)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video bitti.")
        break

    # İzlemesi kolay olsun diye videoyu biraz yavaşlatıyoruz (30ms bekleme)
    cv2.imshow("Jackson Hole - Veri Toplama (Lokal)", frame)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('s'):
        count += 1
        img_name = os.path.join(DATASET_DIR, f"frame_{count:03d}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"📸 Kaydedildi: {img_name}")

    elif key == ord('q'):
        print("İşlem sonlandırıldı.")
        break

cap.release()
cv2.destroyAllWindows()