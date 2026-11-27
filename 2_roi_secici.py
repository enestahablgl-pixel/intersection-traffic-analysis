import cv2
import numpy as np

# ==========================================
# AYARLAR
# ==========================================
VIDEO_PATH = r"D:\staj kod\task3\gunduz_video.mp4" 

cap = cv2.VideoCapture(VIDEO_PATH)

# ==========================================
# VİDEOYU 35. SANİYEYE ATLATMA (YENİ KISIM)
# ==========================================
# 35 saniye = 35000 milisaniye
cap.set(cv2.CAP_PROP_POS_MSEC, 35000)

ret, frame = cap.read()

if not ret:
    print("❌ HATA: Videonun 35. saniyesi okunamadı! Video daha kısa olabilir mi?")
    exit()

all_rois = [] # Tüm yaya geçitlerini tutacak liste
current_points = [] # O an çizilen yaya geçidinin noktaları

# Renk paleti (3 farklı geçit için 3 farklı renk)
COLORS = [(0, 255, 0), (255, 255, 0), (0, 255, 255)] 

def click_event(event, x, y, flags, params):
    global current_points, frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append([x, y])
        roi_idx = len(all_rois) # Şu an kaçıncı yaya geçidini çiziyoruz?
        
        # Tıklanan yere nokta koy
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        
        # Çizgileri birleştir
        if len(current_points) >= 2:
            cv2.line(frame, tuple(current_points[-2]), tuple(current_points[-1]), COLORS[roi_idx], 2)
            
        cv2.imshow("3'lu ROI Secici", frame)
        
        # 4 nokta tamamlandığında o yaya geçidini kaydet
        if len(current_points) == 4:
            cv2.line(frame, tuple(current_points[-1]), tuple(current_points[0]), COLORS[roi_idx], 2)
            cv2.imshow("3'lu ROI Secici", frame)
            
            all_rois.append(current_points)
            print(f"✅ {roi_idx + 1}. Yaya Geçidi Alanı Seçildi!")
            
            current_points = [] # Yeni geçit için sıfırla
            
            # 3 yaya geçidi de bittiyse koordinatları dök
            if len(all_rois) == 3:
                print("\n" + "="*60)
                print("🎉 TÜM YAYA GEÇİTLERİ BAŞARIYLA TANIMLANDI!")
                print("Lütfen aşağıdaki satırları kopyalayıp ana koda yapıştır:\n")
                for i, roi in enumerate(all_rois):
                    print(f"ROI_{i+1} = np.array({roi}, np.int32)")
                print("="*60 + "\n")
                print("Çıkmak için ekrandaki resim üzerindeyken herhangi bir tuşa bas.")

# ==========================================
# PENCERE AYARLARI
# ==========================================
cv2.namedWindow("3'lu ROI Secici", cv2.WINDOW_NORMAL)
cv2.resizeWindow("3'lu ROI Secici", 1280, 720) 

cv2.imshow("3'lu ROI Secici", frame)
cv2.setMouseCallback("3'lu ROI Secici", click_event)

print("KULLANIM KILAVUZU:")
print("1️⃣ ÖNCE 1. yaya geçidinin 4 köşesine sırayla tıkla.")
print("2️⃣ SONRA 2. yaya geçidinin 4 köşesine sırayla tıkla.")
print("3️⃣ EN SON 3. yaya geçidinin 4 köşesine sırayla tıkla.")
print("🚨 Önemli: Her geçidi kendi içinde sırayla (saat yönünde) tıkla, çapraz gitme!")

cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()