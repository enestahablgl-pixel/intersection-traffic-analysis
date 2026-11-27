import cv2
import numpy as np
import os
import time
import torch
import sqlite3
from collections import defaultdict
from ultralytics import YOLO

# ==========================================
# 1. AYARLAR VE ÖZEL TAKİPÇİ (TRACKER)
# ==========================================
VIDEO_PATH = r"D:\staj kod\task3\gunduz_video.mp4"
OUTPUT_PATH = r"D:\staj kod\task3\islenmis_video.mp4"
DEBUG_DIR = r"D:\staj kod\task3\debug_goruntuleri"
DB_PATH = r"D:\staj kod\task3\arac_verileri.db"

os.makedirs(DEBUG_DIR, exist_ok=True)

TRACKER_CONFIG = "custom_tracker.yaml"
with open(TRACKER_CONFIG, "w") as f:
    f.write("""
tracker_type: bytetrack
track_high_thresh: 0.3
track_low_thresh: 0.1
new_track_thresh: 0.4
track_buffer: 90
match_thresh: 0.8
fuse_score: True
""")

# ==========================================
# VERİTABANI
# ==========================================
def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS arac_gecisleri')
    cursor.execute('DROP TABLE IF EXISTS genel_ozet')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS arac_gecisleri (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            arac_id INTEGER,
            yon TEXT,
            gecis_zamani TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS genel_ozet (
            id INTEGER PRIMARY KEY,
            yapay_zeka_ham_id_sayisi INTEGER DEFAULT 0,
            gecerli_gecis_yapan_arac INTEGER DEFAULT 0
        )
    ''')
    cursor.execute("INSERT INTO genel_ozet (id, yapay_zeka_ham_id_sayisi, gecerli_gecis_yapan_arac) VALUES (1, 0, 0)")
    conn.commit()
    return conn, cursor

conn, cursor = init_database()

# ==========================================
# GEOMETRİK ALANLAR (ROI) & RENK PALETİ
# ==========================================
ROI_DUZ = np.array([[910, 652], [1875, 678], [1874, 650], [1006, 628]], np.int32)
ROI_SOL = np.array([[355, 756], [762, 643], [890, 648], [506, 778]], np.int32)
ROI_YAYA_ALT = np.array([[1889, 1027], [396, 825], [532, 779], [1888, 925]], np.int32)
ROI_GIRIS = np.array([[1779, 1020], [1761, 1076], [764, 1072], [944, 936]], np.int32)

YAYA_ROIS = [ROI_DUZ, ROI_SOL, ROI_YAYA_ALT]

RENK_PALETI = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), 
    (255, 0, 255), (255, 255, 0), (0, 165, 255), (147, 20, 255),
    (255, 144, 30), (0, 128, 0)
]

# ==========================================
# 2. MODEL YÜKLEME 
# ==========================================
print("⏳ Model yükleniyor...")
track_model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = int(cap.get(cv2.CAP_PROP_FPS))

roi_mask_yaya = np.zeros((frame_height, frame_width), dtype=np.uint8) 
for roi in YAYA_ROIS:
    cv2.fillPoly(roi_mask_yaya, [roi], 255)

mask_giris = np.zeros((frame_height, frame_width), dtype=np.uint8)
cv2.fillPoly(mask_giris, [ROI_GIRIS], 255)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, video_fps, (frame_width, frame_height))

cv2.namedWindow("Jackson Hole - Entegre Denetim", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Jackson Hole - Entegre Denetim", 1280, 720)

track_history = defaultdict(lambda: [])
vehicle_states = defaultdict(lambda: {"girdi": False, "hedef": None, "kaydedildi": False})

# MATEMATİKSEL KÜME (Aynı ID asla 2 kere yazılamaz)
gorulen_tum_idler = set() 

sayac_sol = 0
sayac_duz = 0

debug_count = 0
print("\n🚀 Agresif Filtreli ve Detaylı Sayaçlı Sistem Başlatıldı!\n")

# ==========================================
# 3. ANA VIDEO DÖNGÜSÜ
# ==========================================
while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret: break

    raw_frame = frame.copy()

    overlay = frame.copy()
    for roi in YAYA_ROIS:
        cv2.fillPoly(overlay, [roi], (255, 0, 100))
    cv2.fillPoly(overlay, [ROI_GIRIS], (255, 150, 0))
    display_frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    # İYİLEŞTİRME: conf=0.30 (Hayaletleri sil) ve iou=0.3 (İç içe kutuları zorla birleştir)
    results_track = track_model.track(
        raw_frame, 
        persist=True, 
        tracker=TRACKER_CONFIG, 
        classes=[0, 2, 3, 5, 7], 
        verbose=False,
        conf=0.30,          
        imgsz=1280,         
        iou=0.3,            
        agnostic_nms=True   
    )

    if results_track[0].boxes.id is not None:
        v_boxes = results_track[0].boxes.xyxy.cpu().numpy().astype(int)
        v_ids = results_track[0].boxes.id.cpu().numpy().astype(int)
        v_cls = results_track[0].boxes.cls.cpu().numpy().astype(int) 

        for box, track_id, class_id in zip(v_boxes, v_ids, v_cls):
            vx1, vy1, vx2, vy2 = box
            vx1_s, vy1_s = max(0, vx1), max(0, vy1)
            vx2_s, vy2_s = min(frame_width, vx2), min(frame_height, vy2)

            if class_id == 0:
                kutu_ici_maske = roi_mask_yaya[vy1_s:vy2_s, vx1_s:vx2_s]
                
                if cv2.countNonZero(kutu_ici_maske) > 0:
                    cv2.rectangle(display_frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 3)
                    cv2.putText(display_frame, "Gecitte", (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.rectangle(display_frame, (vx1, vy1), (vx2, vy2), (0, 0, 255), 3)
                    cv2.putText(display_frame, "Disarida", (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                continue 

            cx = int((vx1 + vx2) / 2)
            alt_y = int(vy2)
            alt_merkez_nokta = (int(cx), int(alt_y))

            # Yeni tespit edilen ID'leri kümeye ekle
            if track_id not in gorulen_tum_idler:
                gorulen_tum_idler.add(track_id)
                cursor.execute("UPDATE genel_ozet SET yapay_zeka_ham_id_sayisi = ? WHERE id = 1", (len(gorulen_tum_idler),))

            arac_rengi = RENK_PALETI[track_id % len(RENK_PALETI)]

            cv2.rectangle(display_frame, (vx1, vy1), (vx2, vy2), (255, 255, 255), 2)
            cv2.putText(display_frame, f"ID:{track_id}", (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(display_frame, alt_merkez_nokta, 6, arac_rengi, -1)

            kutu_giris = mask_giris[vy1_s:vy2_s, vx1_s:vx2_s]
            temas_giris = cv2.countNonZero(kutu_giris) > 100

            if temas_giris:
                vehicle_states[track_id]["girdi"] = True

            if vehicle_states[track_id]["girdi"] and not vehicle_states[track_id]["kaydedildi"]:
                
                track_history[track_id].append((cx, int((vy1+vy2)/2)))
                pts = np.array(track_history[track_id]).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], isClosed=False, color=arac_rengi, thickness=3)

                temas_sol_nokta = cv2.pointPolygonTest(ROI_SOL, alt_merkez_nokta, False) >= 0
                temas_duz_nokta = cv2.pointPolygonTest(ROI_DUZ, alt_merkez_nokta, False) >= 0

                if vehicle_states[track_id]["hedef"] is None:
                    if temas_sol_nokta:
                        vehicle_states[track_id]["hedef"] = "Sola Donus"
                    elif temas_duz_nokta:
                        vehicle_states[track_id]["hedef"] = "Duz Gidis"

                hedef = vehicle_states[track_id]["hedef"]
                cikis_yapti = False
                
                if status := (hedef == "Sola Donus" and not temas_sol_nokta):
                    cikis_yapti = True
                elif status := (hedef == "Duz Gidis" and not temas_duz_nokta):
                    cikis_yapti = True

                if cikis_yapti:
                    if hedef == "Sola Donus":
                        sayac_sol += 1
                    elif hedef == "Duz Gidis":
                        sayac_duz += 1

                    cursor.execute("INSERT INTO arac_gecisleri (arac_id, yon) VALUES (?, ?)", (int(track_id), hedef))
                    cursor.execute("UPDATE genel_ozet SET gecerli_gecis_yapan_arac = gecerli_gecis_yapan_arac + 1 WHERE id = 1")
                    conn.commit()
                    
                    vehicle_states[track_id]["kaydedildi"] = True
                    track_history[track_id] = [] 

    end_time = time.time()
    
    # ==========================================
    # CANLI EKRAN PANOSU (DASHBOARD)
    # ==========================================
    toplam_onaylanan = sayac_sol + sayac_duz
    uretilen_ham_id = len(gorulen_tum_idler)

    cv2.putText(display_frame, f"FPS: {int(1 / (end_time - start_time))}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    cv2.putText(display_frame, f"Sola Giden: {sayac_sol}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Duz Giden: {sayac_duz}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(display_frame, f"Onaylanan Toplam Arac: {toplam_onaylanan}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Ham ID'yi daha soluk bir renkle alta ekledik ki asıl önemli sayılarla karışmasın
    cv2.putText(display_frame, f"Uretilen Ham ID: {uretilen_ham_id}", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    out.write(display_frame)
    cv2.imshow("Jackson Hole - Entegre Denetim", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break

cap.release()
out.release()
conn.close()
cv2.destroyAllWindows()