"""Araç + plaka çıkarımı, kutu eşleştirme ve track başına OCR.

Bu dosyanın SAF kısmı (match_plates_to_vehicles + yardımcılar + config) ağır
bağımlılık içermez ve testlerde modelsiz kullanılır. Model kısmı (InferenceEngine)
ultralytics/easyocr'ı YALNIZCA fonksiyon içinde (lazy) import eder; böylece saf
fonksiyonlar torch olmadan import edilebilir.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from server.session import Reading, SessionStore, vote

# COCO sınıf indeksleri -> araç sınıf adı (yalnızca bu 4'ü kullanılır)
COCO_VEHICLE_CLASSES: Dict[int, str] = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class InferenceConfig:
    """Çıkarım ayarları. Adım 5'te config.example.yaml'dan doldurulur."""

    vehicle_conf: float = 0.35
    plate_conf: float = 0.25
    ocr_min_conf: float = 0.0
    vote_window: int = 8
    track_max_age: float = 5.0
    max_cameras: int = 4
    max_ocr_per_frame: int = 3
    vehicle_weights: str = "yolov8n.pt"       # COCO (yeni eğitim yok)
    plate_weights: str = "weights/best.pt"    # mevcut plaka modeli


@dataclass
class PlateDet:
    """Bir karedeki plaka tespiti (kutu + tespit güveni)."""

    box: List[int]   # [x1, y1, x2, y2] piksel
    conf: float


@dataclass
class MatchResult:
    """Plaka->araç eşleştirme sonucu."""

    # araç indeksi -> o araca ait BİRİNCİL plaka (en yüksek conf)
    vehicle_plate: Dict[int, PlateDet]
    # aynı araca atanmış fazladan plakalar (tır: çekici+dorse). Şu an cevaba
    # DAHİL EDİLMEZ; ileride dorse takibi için ayrılır.
    secondary_plates: List[PlateDet] = field(default_factory=list)
    # hiçbir aracın içinde olmayan plakalar (cevaptaki orphan_plates)
    orphan_plates: List[PlateDet] = field(default_factory=list)


def _area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _center(b: Sequence[float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _inside(point: tuple[float, float], box: Sequence[float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def match_plates_to_vehicles(
    vehicle_boxes: List[Sequence[int]], plates: List[PlateDet]
) -> MatchResult:
    """Plakaları araçlara MERKEZ-içerme kuralıyla eşleştirir. SAF fonksiyon.

    Kurallar:
      - Plaka kutusunun MERKEZİ hangi araç kutusunun içindeyse o araca atanır.
      - Merkez birden fazla araç kutusunun içindeyse EN KÜÇÜK ALANLI araç seçilir
        (eşit alanda: küçük araç indeksi — deterministik).
      - Merkez hiçbir aracın içinde değilse plaka orphan_plates'e gider.
      - Bir araca birden fazla plaka atanırsa (tır: çekici + dorse) EN YÜKSEK
        conf'lu plaka birincil (vehicle.plate) olur; diğerleri orphan'a DEĞİL,
        secondary_plates'e gider ve şu an cevaba dahil edilmez (ileride dorse
        takibi için). conf eşitliğinde giriş sırası küçük olan birincil olur.

    Model, kamera, torch bilmez; yalnızca kutu listeleri alır.
    """
    assigned: Dict[int, List[tuple[int, PlateDet]]] = defaultdict(list)
    orphans: List[PlateDet] = []

    for p_idx, plate in enumerate(plates):
        c = _center(plate.box)
        containing = [v for v, vbox in enumerate(vehicle_boxes) if _inside(c, vbox)]
        if not containing:
            orphans.append(plate)
            continue
        v_idx = min(containing, key=lambda v: (_area(vehicle_boxes[v]), v))
        assigned[v_idx].append((p_idx, plate))

    vehicle_plate: Dict[int, PlateDet] = {}
    secondary: List[PlateDet] = []
    for v_idx, plist in assigned.items():
        ranked = sorted(plist, key=lambda t: (-t[1].conf, t[0]))
        vehicle_plate[v_idx] = ranked[0][1]
        secondary.extend(t[1] for t in ranked[1:])

    return MatchResult(vehicle_plate=vehicle_plate,
                       secondary_plates=secondary,
                       orphan_plates=orphans)


# ----------------------------------------------------------------------------
# MODEL KISMI — ultralytics/easyocr YALNIZCA fonksiyon içinde (lazy) import edilir
# ----------------------------------------------------------------------------
class InferenceEngine:
    """Araç+plaka çıkarımı, per-camera tracker izolasyonu, track başına OCR+oy.

    HTTP/FastAPI/asyncio BİLMEZ (o Adım 5). infer() InferResponse'a doğrudan
    dönüştürülebilen bir dict döndürür (vehicle sınıfı 'class' anahtarıyla).
    """

    def __init__(self, config: Optional[InferenceConfig] = None):
        self.cfg = config or InferenceConfig()
        self._plate = None
        self._vehicle_models: Dict[str, object] = {}
        self._store = SessionStore(vote_window=self.cfg.vote_window)
        self._frame_idx: Dict[str, int] = defaultdict(int)

    # --- model yükleme (lazy) ---
    def _log_gpu_mem(self, msg: str) -> None:
        reserved = self.gpu_mem_reserved_mb()
        if reserved is None:
            print(f"[inference] {msg} | cihaz: cpu", flush=True)
        else:
            print(f"[inference] {msg} | GPU reserved: {reserved:.0f} MB", flush=True)

    def _plate_model(self):
        if self._plate is None:
            from ultralytics import YOLO
            self._plate = YOLO(self.cfg.plate_weights)
            self._log_gpu_mem("plaka modeli yüklendi")
        return self._plate

    def _vehicle_model(self, camera_id: str):
        if camera_id in self._vehicle_models:
            return self._vehicle_models[camera_id]
        if len(self._vehicle_models) >= self.cfg.max_cameras:
            raise RuntimeError(
                f"max_cameras={self.cfg.max_cameras} sınırı aşıldı; "
                f"yeni kamera '{camera_id}' için model YÜKLENMEDİ.")
        from ultralytics import YOLO
        model = YOLO(self.cfg.vehicle_weights)
        self._vehicle_models[camera_id] = model
        self._log_gpu_mem(f"araç modeli yüklendi (camera={camera_id}, "
                          f"toplam kamera={len(self._vehicle_models)})")
        return model

    def device(self) -> str:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def gpu_mem_reserved_mb(self) -> Optional[float]:
        """torch.cuda.memory_reserved() (MB). cuDNN workspace ilk forward pass'te
        ayrıldığı için anlamlı değer WARMUP'TAN SONRA okunur."""
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / 1e6
        return None

    def active_cameras(self) -> List[str]:
        """Model yüklenmiş kamera kimlikleri."""
        return list(self._vehicle_models.keys())

    def models_loaded(self) -> bool:
        return self._plate is not None

    def warmup(self) -> None:
        """Başlangıçta modelleri yükle ve sahte kareyle bir kez çalıştır.

        Kamera slotu TÜKETMEZ: araç modeli geçici olarak yüklenip çalıştırılır ve
        bırakılır (CUDA/cuDNN yolu ısınır, allocator bellek tutar). EasyOCR de
        ısıtılır ki ilk gerçek istek beklemesin.
        """
        import numpy as np
        blank = np.zeros((640, 640, 3), dtype=np.uint8)

        self._plate_model().predict(blank, conf=self.cfg.plate_conf, verbose=False)

        from ultralytics import YOLO
        tmp = YOLO(self.cfg.vehicle_weights)
        tmp.predict(blank, classes=[2, 3, 5, 7], conf=self.cfg.vehicle_conf,
                    verbose=False)
        del tmp

        from ocr import get_reader, preprocess
        from shared.plate_format import ALLOWLIST
        reader = get_reader()
        reader.readtext(preprocess(np.zeros((40, 160, 3), dtype=np.uint8)),
                        allowlist=ALLOWLIST, detail=0, paragraph=False)

        self._log_gpu_mem("warmup tamamlandı")

    # --- OCR: ocr.py yolu (preprocess + reader); conf için detail=1 ---
    def _ocr_read(self, crop):
        import numpy as np
        from ocr import get_reader, preprocess
        from shared.plate_format import ALLOWLIST, format_plate
        reader = get_reader()
        img = preprocess(crop)
        parts = reader.readtext(img, allowlist=ALLOWLIST, detail=1, paragraph=False)
        raw = re.sub(r"[^0-9A-Z]", "", "".join(t for _, t, _ in parts).upper())
        ocr_conf = float(np.mean([c for _, _, c in parts])) if parts else 0.0
        text = format_plate(raw)
        valid = text is not None and ocr_conf >= self.cfg.ocr_min_conf
        return raw, (text or ""), valid, ocr_conf

    def _need_ocr(self, existing) -> bool:
        """OCR tetikleme kuralı (GPU'yu korumak için)."""
        valid_count = sum(1 for r in existing if r.valid)
        if valid_count == 0:
            return True
        if valid_count < self.cfg.vote_window:
            return True
        v = vote(existing)                       # margin==0 -> beraberlik, kararsız
        return v is not None and v.margin == 0

    # --- ana giriş ---
    def infer(self, frame, camera_id: str, frame_ts: int) -> dict:
        import time
        t0 = time.perf_counter()
        h, w = int(frame.shape[0]), int(frame.shape[1])
        ts_sec = frame_ts / 1000.0

        # eskiyen tracklerı temizle (bellek sızıntısı olmasın)
        self._store.evict_stale(now_ts=ts_sec, max_age_sec=self.cfg.track_max_age)

        # 1) Araç tespiti + takip (tam kare, COCO sınıf filtresi)
        vmodel = self._vehicle_model(camera_id)
        vres = vmodel.track(frame, persist=True, classes=[2, 3, 5, 7],
                            conf=self.cfg.vehicle_conf, tracker="bytetrack.yaml",
                            verbose=False)[0]
        vehicle_boxes: List[List[int]] = []
        vehicle_meta: List[dict] = []
        vb = vres.boxes
        if vb is not None and len(vb):
            xyxy = vb.xyxy.cpu().numpy()
            clss = vb.cls.cpu().numpy().astype(int)
            confs = vb.conf.cpu().numpy()
            ids = vb.id.cpu().numpy().astype(int) if vb.id is not None else None
            for i in range(len(xyxy)):
                vehicle_boxes.append([int(x) for x in xyxy[i]])
                vehicle_meta.append({
                    "class": COCO_VEHICLE_CLASSES.get(int(clss[i]), "car"),
                    "conf": float(confs[i]),
                    "track_id": int(ids[i]) if ids is not None else None,
                })

        # 2) Plaka tespiti — AYNI TAM KARE (aracın kırpığında DEĞİL)
        pmodel = self._plate_model()
        pres = pmodel.predict(frame, conf=self.cfg.plate_conf, verbose=False)[0]
        plates: List[PlateDet] = []
        pb = pres.boxes
        if pb is not None and len(pb):
            pxyxy = pb.xyxy.cpu().numpy()
            pconf = pb.conf.cpu().numpy()
            for i in range(len(pxyxy)):
                plates.append(PlateDet(box=[int(x) for x in pxyxy[i]],
                                       conf=float(pconf[i])))

        # 3) Eşleştirme
        match = match_plates_to_vehicles(vehicle_boxes, plates)

        # görülen tracklerin son görülme zamanını tazele
        for meta in vehicle_meta:
            if meta["track_id"] is not None:
                self._store.touch(camera_id, meta["track_id"], ts_sec)

        # 4) OCR bütçesi: OCR gereken (tracked + plakalı) araçları seç
        candidates = []
        for v_idx, meta in enumerate(vehicle_meta):
            tid = meta["track_id"]
            plate_det = match.vehicle_plate.get(v_idx)
            if tid is None or plate_det is None:
                continue
            if self._need_ocr(self._store.get_readings(camera_id, tid)):
                candidates.append((plate_det.conf, v_idx))
        candidates.sort(key=lambda t: -t[0])          # yüksek plaka conf önce
        selected = {v for _, v in candidates[:self.cfg.max_ocr_per_frame]}

        frame_idx = self._frame_idx[camera_id]
        self._frame_idx[camera_id] = frame_idx + 1
        for v_idx in selected:
            plate_det = match.vehicle_plate[v_idx]
            tid = vehicle_meta[v_idx]["track_id"]
            x1, y1, x2, y2 = plate_det.box
            crop = frame[max(y1, 0):y2, max(x1, 0):x2]
            if crop.size == 0:
                continue
            raw, text, valid, ocr_conf = self._ocr_read(crop)
            self._store.add(camera_id, tid,
                            Reading(raw=raw, text=text, valid=valid,
                                    ocr_conf=ocr_conf, plate_conf=plate_det.conf,
                                    frame_idx=frame_idx),
                            ts=ts_sec)

        # 5) Cevabı kur (alan kaynakları madde 7'ye göre)
        vehicles_out = []
        for v_idx, meta in enumerate(vehicle_meta):
            tid = meta["track_id"]
            plate_det = match.vehicle_plate.get(v_idx)
            plate_field = None
            if tid is not None and plate_det is not None:
                v = vote(self._store.get_readings(camera_id, tid))
                if v is not None:
                    plate_field = {"box": plate_det.box, "conf": plate_det.conf,
                                   "raw": v.raw, "text": v.text,
                                   "valid": v.valid, "votes": v.votes}
                else:  # plaka tespit edildi ama henüz OCR yok (bütçe)
                    plate_field = {"box": plate_det.box, "conf": plate_det.conf,
                                   "raw": "", "text": "", "valid": False, "votes": 0}
            # tid None (untracked) veya plakasız -> plate_field None
            vehicles_out.append({
                "track_id": tid,
                "class": meta["class"],
                "box": vehicle_boxes[v_idx],
                "conf": meta["conf"],
                "plate": plate_field,
            })

        # orphan plakalar: OCR track-tabanlı olduğundan uygulanmaz -> text daima None
        orphans_out = [{"box": p.box, "conf": p.conf, "text": None}
                       for p in match.orphan_plates]

        return {
            "camera_id": camera_id,
            "frame_ts": frame_ts,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "frame_w": w,
            "frame_h": h,
            "vehicles": vehicles_out,
            "orphan_plates": orphans_out,
        }
