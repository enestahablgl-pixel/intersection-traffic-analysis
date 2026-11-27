import cv2
import numpy as np
import os
import shutil
import time
import torch
import sqlite3
import threading
from collections import defaultdict
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

# ==========================================
# 1. AYARLAR VE ÖZEL TAKİPÇİ (TRACKER)
# ==========================================
VIDEO_PATH = r"D:\staj kod\task3\gunduz_video.mp4"
OUTPUT_PATH = r"D:\staj kod\task3\islenmis_video.mp4"
DEBUG_DIR = r"D:\staj kod\task3\debug_goruntuleri"
DB_PATH = r"D:\staj kod\task3\arac_verileri.db"

# >>> PERFORMANS TEST ANAHTARLARI <<<
# Darboğazı bulmak için bunları tek tek kapatıp/açıp FPS'i karşılaştırın.
SHOW_WINDOW = True          # False yaparsanız cv2.imshow tamamen devre dışı kalır
WRITE_VIDEO = True          # False yaparsanız out.write() maliyeti devre dışı kalır ; True ise artık ASENKRON (ayrı thread) yazar
PRINT_TIMING_EVERY = 30     # Her N karede bir konsola aşama sürelerini yazdırır
SAHI_TARGET_FPS = 10        # Yaya (SAHI) tespiti bu hızın üzerine çıkmaya çalışmaz -> GPU'dan tracking'e pay bırakır
SAHI_ENABLED = True         # <<< TEŞHİS: False yapıp aynı 30-kare testini tekrarlayın.
                            #     tracking süresi düşerse sorun GPU çekişmesi; düşmezse sorun engine/precision'da demektir.
FORCE_REEXPORT_TRACK_ENGINE = False  # <<< Eski/bozuk .engine dosyasından şüpheleniyorsanız bir kerelik True yapıp çalıştırın

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
# VERİTABANI YAPILANDIRMASI
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
# GEOMETRİK ALANLAR (ROI) & RENK PALETI
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
# 2. MODELLERİN YÜKLENMESİ (TensorRT Engine Desteği)
# ==========================================
# >>> DÜZELTME: Önceki versiyonda engine 960 ile export edilip track() çağrısında
# imgsz=1280 veriliyordu -> bu iki değer TUTARLI OLMALI, yoksa TensorRT engine
# ya hata verir ya da Ultralytics sessizce farklı bir davranışa düşer ve
# optimizasyon fiilen hiç uygulanmamış olur. Artık TEK bir yerden (TRACK_IMGSZ)
# yönetiliyor; export, warm-up ve gerçek track() çağrısı hep aynı değeri kullanıyor.
TRACK_IMGSZ = 960
TRACK_ENGINE_PATH = rf"D:\staj kod\task3\yolo11n_track_{TRACK_IMGSZ}.engine"
SAHI_ENGINE_PATH = r"D:\staj kod\task3\yolo11n_sahi_640.engine"

print("⏳ Modeller yükleniyor (SAHI Asenkron/Thread Modu)...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_HALF = torch.cuda.is_available()
USE_TENSORRT = torch.cuda.is_available()


def export_or_load_engine(pt_path, desired_engine_path, imgsz, use_half):
    if os.path.exists(desired_engine_path):
        print(f"✅ Mevcut TensorRT engine bulundu, tekrar export edilmeyecek: {desired_engine_path}")
        return desired_engine_path

    print(f"⏳ TensorRT engine oluşturuluyor (imgsz={imgsz}, half={use_half})... Bu birkaç dakika sürebilir.")
    temp_model = YOLO(pt_path)
    exported_path = str(temp_model.export(format="engine", imgsz=imgsz, half=use_half, device=0))

    os.makedirs(os.path.dirname(desired_engine_path), exist_ok=True)
    if os.path.abspath(exported_path) != os.path.abspath(desired_engine_path):
        shutil.move(exported_path, desired_engine_path)

    print(f"✅ Engine hazır: {desired_engine_path}")
    return desired_engine_path


if USE_TENSORRT:
    try:
        if FORCE_REEXPORT_TRACK_ENGINE and os.path.exists(TRACK_ENGINE_PATH):
            print(f"🗑️ FORCE_REEXPORT_TRACK_ENGINE=True -> eski engine siliniyor: {TRACK_ENGINE_PATH}")
            os.remove(TRACK_ENGINE_PATH)

        track_model_source = export_or_load_engine("yolo11n.pt", TRACK_ENGINE_PATH, imgsz=TRACK_IMGSZ, use_half=USE_HALF)
        sahi_model_source = export_or_load_engine("yolo11n.pt", SAHI_ENGINE_PATH, imgsz=640, use_half=USE_HALF)
    except Exception as e:
        print(f"⚠️ TensorRT export başarısız oldu ({e}). Normal .pt modele geri dönülüyor.")
        track_model_source = "yolo11n.pt"
        sahi_model_source = "yolo11n.pt"
else:
    print("⚠️ CUDA bulunamadı, TensorRT NVIDIA GPU gerektirir. .pt modeline devam ediliyor.")
    track_model_source = "yolo11n.pt"
    sahi_model_source = "yolo11n.pt"

detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=sahi_model_source,
    confidence_threshold=0.25,
    device=device
)

track_model = YOLO(track_model_source)

# >>> TEŞHİS: gerçekten hangi backend/precision çalışıyor, konsola yazdır.
print(f"ℹ️ Track model kaynağı : {track_model_source}")
print(f"ℹ️ Track model backend  : {type(track_model.model).__name__ if hasattr(track_model, 'model') else 'bilinmiyor'}")
print(f"ℹ️ USE_HALF (FP16)      : {USE_HALF}")
if os.path.exists(TRACK_ENGINE_PATH):
    print(f"ℹ️ Engine dosya boyutu  : {os.path.getsize(TRACK_ENGINE_PATH) / 1e6:.1f} MB  (bariz küçükse export yarım kalmış olabilir)")

cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = int(cap.get(cv2.CAP_PROP_FPS))

roi_mask_yaya = np.zeros((frame_height, frame_width), dtype=np.uint8)
for roi in YAYA_ROIS:
    cv2.fillPoly(roi_mask_yaya, [roi], 255)

mask_giris = np.zeros((frame_height, frame_width), dtype=np.uint8)
cv2.fillPoly(mask_giris, [ROI_GIRIS], 255)

# >>> DÜZELTME: ROI'ler statik (video boyunca hiç değişmiyor) ama eski kodda
# overlay (fillPoly + addWeighted) HER KAREDE yeniden hesaplanıyordu. Bunun
# yerine statik renkli katmanı bir kere üretip sadece alfa karışımını her
# karede uyguluyoruz -> fillPoly çağrıları döngüden tamamen çıktı.
STATIC_OVERLAY = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
for roi in YAYA_ROIS:
    cv2.fillPoly(STATIC_OVERLAY, [roi], (255, 0, 100))
cv2.fillPoly(STATIC_OVERLAY, [ROI_GIRIS], (255, 150, 0))
# Overlay'in nerede boyalı olduğunu gösteren maske (addWeighted'i sadece o
# bölgede uygulamak için) - fillPoly olmayan yerlerde orijinal frame kalsın.
STATIC_OVERLAY_MASK = cv2.cvtColor(STATIC_OVERLAY, cv2.COLOR_BGR2GRAY) > 0

# Blend işlemini tüm kareye değil, sadece ROI'leri kapsayan küçük dikdörtgene
# uygulamak için sınır kutusu (bounding box) hesaplıyoruz.
_all_roi_pts = np.vstack(YAYA_ROIS + [ROI_GIRIS])
OVERLAY_BX1 = max(0, int(_all_roi_pts[:, 0].min()) - 10)
OVERLAY_BY1 = max(0, int(_all_roi_pts[:, 1].min()) - 10)
OVERLAY_BX2 = min(frame_width, int(_all_roi_pts[:, 0].max()) + 10)
OVERLAY_BY2 = min(frame_height, int(_all_roi_pts[:, 1].max()) + 10)

# ==========================================
# SAHI'Yİ SADECE YAYA ROI BÖLGESİNE UYGULA
# ==========================================
def yaya_crop_bolgesini_hesapla(rois, padding=60):
    all_pts = np.vstack(rois)
    x1 = max(0, int(all_pts[:, 0].min()) - padding)
    y1 = max(0, int(all_pts[:, 1].min()) - padding)
    x2 = min(frame_width, int(all_pts[:, 0].max()) + padding)
    y2 = min(frame_height, int(all_pts[:, 1].max()) + padding)
    return x1, y1, x2, y2

YAYA_CROP_X1, YAYA_CROP_Y1, YAYA_CROP_X2, YAYA_CROP_Y2 = yaya_crop_bolgesini_hesapla(YAYA_ROIS)
print(f"ℹ️ SAHI şu bölgede çalışacak: ({YAYA_CROP_X1},{YAYA_CROP_Y1})-({YAYA_CROP_X2},{YAYA_CROP_Y2})  "
      f"[{YAYA_CROP_X2 - YAYA_CROP_X1}x{YAYA_CROP_Y2 - YAYA_CROP_Y1} px]")

# ==========================================
# ISINMA (WARM-UP) — artık TRACK_IMGSZ ile tutarlı
# ==========================================
print("🔥 Modeller ısıtılıyor (warm-up)...")
dummy_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

track_model.predict(dummy_frame, imgsz=TRACK_IMGSZ, verbose=False, half=USE_HALF)

_ = get_sliced_prediction(
    dummy_frame, detection_model,
    slice_height=640, slice_width=640,
    overlap_height_ratio=0.1, overlap_width_ratio=0.1, verbose=0
)
print("✅ Isınma tamamlandı.")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, video_fps, (frame_width, frame_height)) if WRITE_VIDEO else None

# >>> ASENKRON YAZMA: out.write() artık ana döngüyü beklemesin diye ayrı thread'e alındı.
import queue
write_queue = queue.Queue(maxsize=64)
stop_writer_thread = False


def writer_worker():
    while True:
        try:
            item = write_queue.get(timeout=0.5)
        except queue.Empty:
            if stop_writer_thread:
                break
            continue
        if item is None:
            break
        out.write(item)


writer_thread = None
if WRITE_VIDEO:
    writer_thread = threading.Thread(target=writer_worker, daemon=True)
    writer_thread.start()

if SHOW_WINDOW:
    cv2.namedWindow("Jackson Hole - Entegre Denetim", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Jackson Hole - Entegre Denetim", 1280, 720)

track_history = defaultdict(lambda: [])
vehicle_states = defaultdict(lambda: {"girdi": False, "hedef": None, "kaydedildi": False})
gorulen_tum_idler = set()

# ==========================================
# ASENKRON SAHI (YAYA TESPİTİ) ALT YAPISI — artık throttle'lı
# ==========================================
sahi_lock = threading.Lock()
frame_lock = threading.Lock()

last_yaya_predictions = []
latest_raw_frame = None
sahi_last_update_time = 0.0
stop_sahi_thread = False

SAHI_MIN_INTERVAL = 1.0 / SAHI_TARGET_FPS


def sahi_worker():
    global last_yaya_predictions, latest_raw_frame, sahi_last_update_time

    while not stop_sahi_thread:
        loop_start = time.time()

        with frame_lock:
            frame_to_process = None if latest_raw_frame is None else latest_raw_frame.copy()

        if frame_to_process is None:
            time.sleep(0.005)
            continue

        yaya_crop = frame_to_process[YAYA_CROP_Y1:YAYA_CROP_Y2, YAYA_CROP_X1:YAYA_CROP_X2]

        result_sahi = get_sliced_prediction(
            yaya_crop, detection_model,
            slice_height=640, slice_width=640,
            overlap_height_ratio=0.1, overlap_width_ratio=0.1, verbose=0
        )

        with sahi_lock:
            last_yaya_predictions = result_sahi.object_prediction_list
            sahi_last_update_time = time.time()

        # >>> THROTTLE: SAHI'yi kasıtlı olarak SAHI_TARGET_FPS ile sınırlıyoruz.
        # Yaya "geçitte mi" bilgisi 30 kez/sn tazelenmek zorunda değil; bu boşta
        # kalan süre GPU'yu tracking modeline bırakır.
        elapsed = time.time() - loop_start
        remaining = SAHI_MIN_INTERVAL - elapsed
        if remaining > 0:
            time.sleep(remaining)


sahi_thread = threading.Thread(target=sahi_worker, daemon=True)
if SAHI_ENABLED:
    sahi_thread.start()
else:
    print("⚠️ SAHI_ENABLED=False -> yaya tespiti tamamen devre dışı (teşhis modu).")

frame_count = 0
sayac_sol = 0
sayac_duz = 0

# Aşama bazlı zamanlama toplamları (debug için)
timing_acc = defaultdict(float)
timing_frames = 0

print("\n🚀 Optimize Edilmiş (Timing + Throttle) Versiyon Başlatıldı!\n")

# ==========================================
# 3. ANA VIDEO DÖNGÜSÜ
# ==========================================
while cap.isOpened():
    loop_t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    t_read = time.time()

    frame_count += 1
    _t_copy0 = time.time()
    raw_frame = frame.copy()
    _t_copy1 = time.time()

    # >>> DÜZELTME: fillPoly artık döngüde yok; addWeighted de artık tüm kareye
    # değil sadece ROI'leri kapsayan küçük dikdörtgene uygulanıyor (çok daha ucuz).
    display_frame = frame.copy()
    bx1, by1, bx2, by2 = OVERLAY_BX1, OVERLAY_BY1, OVERLAY_BX2, OVERLAY_BY2
    sub_frame = frame[by1:by2, bx1:bx2]
    sub_overlay = STATIC_OVERLAY[by1:by2, bx1:bx2]
    sub_mask = STATIC_OVERLAY_MASK[by1:by2, bx1:bx2]
    sub_blended = cv2.addWeighted(sub_overlay, 0.3, sub_frame, 0.7, 0)
    sub_display = display_frame[by1:by2, bx1:bx2]
    sub_display[sub_mask] = sub_blended[sub_mask]
    _t_overlay1 = time.time()

    with frame_lock:
        latest_raw_frame = raw_frame

    with sahi_lock:
        yaya_predictions_to_draw = last_yaya_predictions
        sahi_gecikme_ms = int((time.time() - sahi_last_update_time) * 1000) if sahi_last_update_time > 0 else 0

    for obj in yaya_predictions_to_draw:
        if obj.category.id != 0:
            continue

        x1, y1, x2, y2 = map(int, obj.bbox.to_xyxy())
        x1 += YAYA_CROP_X1; x2 += YAYA_CROP_X1
        y1 += YAYA_CROP_Y1; y2 += YAYA_CROP_Y1

        x1_safe, y1_safe = max(0, x1), max(0, y1)
        x2_safe, y2_safe = min(frame_width, x2), min(frame_height, y2)
        kutu_ici_maske = roi_mask_yaya[y1_safe:y2_safe, x1_safe:x2_safe]

        if cv2.countNonZero(kutu_ici_maske) > 0:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(display_frame, "Gecitte", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(display_frame, "Disarida", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    t_sahi_draw = time.time()

    # ---------------------------------------------------------
    # FAZ 2: ARAÇ TAKİBİ — artık TRACK_IMGSZ ile export edilen engine tutarlı
    # ---------------------------------------------------------
    results_track = track_model.track(
        raw_frame,
        persist=True,
        tracker=TRACKER_CONFIG,
        classes=[2, 3, 5, 7],
        verbose=False,
        conf=0.30,
        imgsz=TRACK_IMGSZ,
        iou=0.3,
        agnostic_nms=True,
        half=USE_HALF
    )
    t_track = time.time()

    if results_track[0].boxes.id is not None:
        v_boxes = results_track[0].boxes.xyxy.cpu().numpy().astype(int)
        v_ids = results_track[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(v_boxes, v_ids):
            vx1, vy1, vx2, vy2 = box
            cx = int((vx1 + vx2) / 2)
            alt_y = int(vy2)
            alt_merkez_nokta = (int(cx), int(alt_y))

            if track_id not in gorulen_tum_idler:
                gorulen_tum_idler.add(track_id)
                cursor.execute("UPDATE genel_ozet SET yapay_zeka_ham_id_sayisi = ? WHERE id = 1", (len(gorulen_tum_idler),))

            arac_rengi = RENK_PALETI[track_id % len(RENK_PALETI)]

            cv2.rectangle(display_frame, (vx1, vy1), (vx2, vy2), (255, 255, 255), 2)
            cv2.putText(display_frame, f"ID:{track_id}", (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(display_frame, alt_merkez_nokta, 6, arac_rengi, -1)

            vx1_s, vy1_s = max(0, vx1), max(0, vy1)
            vx2_s, vy2_s = min(frame_width, vx2), min(frame_height, vy2)
            kutu_giris = mask_giris[vy1_s:vy2_s, vx1_s:vx2_s]
            temas_giris = cv2.countNonZero(kutu_giris) > 100

            if temas_giris:
                vehicle_states[track_id]["girdi"] = True

            if vehicle_states[track_id]["girdi"] and not vehicle_states[track_id]["kaydedildi"]:

                track_history[track_id].append((cx, int((vy1 + vy2) / 2)))
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

    t_logic = time.time()

    # ==========================================
    # CANLI DASHBOARD
    # ==========================================
    toplam_onaylanan = sayac_sol + sayac_duz
    uretilen_ham_id = len(gorulen_tum_idler)
    anlik_fps = 1 / (time.time() - loop_t0 + 1e-6)

    cv2.putText(display_frame, f"FPS: {int(anlik_fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    cv2.putText(display_frame, f"Sola Giden: {sayac_sol}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Duz Giden: {sayac_duz}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(display_frame, f"Onaylanan Toplam Arac: {toplam_onaylanan}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(display_frame, f"Uretilen Ham ID: {uretilen_ham_id}", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(display_frame, f"Yaya Tespit Gecikmesi (SAHI): {sahi_gecikme_ms} ms", (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

    if WRITE_VIDEO:
        try:
            write_queue.put_nowait(display_frame.copy())
        except queue.Full:
            pass  # disk yetişemiyorsa kare atla, ana döngüyü bekletme
    t_write = time.time()

    if SHOW_WINDOW:
        cv2.imshow("Jackson Hole - Entegre Denetim", display_frame)
    t_show = time.time()

    # ---- Aşama bazlı zamanlama biriktirme ----
    timing_acc["okuma"] += (t_read - loop_t0)
    timing_acc["  ->kopyalama"] += (_t_copy1 - _t_copy0)
    timing_acc["  ->overlay"] += (_t_overlay1 - _t_copy1)
    timing_acc["sahi_cizim"] += (t_sahi_draw - t_read)
    timing_acc["tracking"] += (t_track - t_sahi_draw)
    timing_acc["mantik"] += (t_logic - t_track)
    timing_acc["yazma"] += (t_write - t_logic)
    timing_acc["gosterim"] += (t_show - t_write)
    timing_frames += 1

    if timing_frames >= PRINT_TIMING_EVERY:
        toplam = sum(timing_acc.values())
        print(f"--- Ortalama aşama süreleri (son {timing_frames} kare, toplam {toplam/timing_frames*1000:.1f} ms/kare) ---")
        for asama, sure in timing_acc.items():
            print(f"   {asama:12s}: {sure/timing_frames*1000:6.1f} ms  ({sure/toplam*100:4.1f}%)")
        timing_acc.clear()
        timing_frames = 0

    key = cv2.waitKey(1) & 0xFF if SHOW_WINDOW else -1
    if key == ord('q'):
        break

# ==========================================
# TEMİZLİK
# ==========================================
stop_sahi_thread = True
if SAHI_ENABLED:
    sahi_thread.join(timeout=2.0)

if WRITE_VIDEO:
    stop_writer_thread = True
    write_queue.put(None)
    writer_thread.join(timeout=5.0)

cap.release()
if WRITE_VIDEO:
    out.release()
conn.close()
if SHOW_WINDOW:
    cv2.destroyAllWindows()