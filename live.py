"""Canlı kamera ile plaka tespiti + OCR (yerel OpenCV penceresi).

Kamerayı açar, her karede plakayı YOLOv8 ile tespit edip kutular (canlı),
OCR'ı belirli aralıklarla (varsayılan 1 sn) çalıştırıp okunan plakayı yazar.
OCR ağır olduğu için her kare yerine aralıklı çalışır; tespit ise gerçek zamanlı.

Kullanım:
    python live.py                     # varsayılan kamera (0)
    python live.py --camera 1          # başka bir kamera
    python live.py --source video.mp4  # kamera yerine video dosyası
    python live.py --conf 0.3 --ocr-interval 0.7

Tuşlar:  q = çıkış,  r = hemen oku (OCR'ı zorla)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from ocr import read_plate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Canlı plaka tespiti + OCR")
    p.add_argument("--source", default="0",
                   help="Kamera indeksi (0, 1, ...) veya video dosyası yolu")
    p.add_argument("--camera", type=int, default=None,
                   help="Kamera indeksi (--source yerine kısayol)")
    p.add_argument("--weights", type=Path, default=Path("weights/best.pt"))
    p.add_argument("--conf", type=float, default=0.25, help="Tespit güven eşiği")
    p.add_argument("--ocr-interval", type=float, default=1.0,
                   help="OCR'ın kaç saniyede bir çalışacağı")
    return p.parse_args()


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    if args.camera is not None:
        return cv2.VideoCapture(args.camera)
    if str(args.source).isdigit():
        return cv2.VideoCapture(int(args.source))
    return cv2.VideoCapture(str(args.source))


def draw_label(frame, box, text: str, color) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)


def nearest_text(box, reads):
    """Geçmiş OCR sonuçlarından bu kutuya en yakın metni bulur (kutu hareket
    etse de metin takip etsin diye)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    best, best_d = None, 1e9
    for rcx, rcy, text in reads:
        d = ((cx - rcx) ** 2 + (cy - rcy) ** 2) ** 0.5
        if d < best_d:
            best, best_d = text, d
    return best if best_d < 150 else None


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise SystemExit(f"HATA: {args.weights} yok. Önce Aşama 1 modelini eğitin.")

    model = YOLO(args.weights)
    cap = open_capture(args)
    if not cap.isOpened():
        raise SystemExit("HATA: Kamera/görüntü kaynağı açılamadı. "
                         "Kamera indeksini (--camera 1) veya dosya yolunu kontrol edin.")

    print("Canlı okuma başladı. Çıkış: 'q', hemen oku: 'r'")
    reads: list[tuple[float, float, str]] = []
    last_ocr = 0.0
    prev = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(source=frame, conf=args.conf, verbose=False)[0]
        boxes = (result.boxes.xyxy.cpu().numpy().astype(int)
                 if result.boxes is not None else [])

        now = time.time()
        if len(boxes) and now - last_ocr >= args.ocr_interval:
            reads = []
            for x1, y1, x2, y2 in boxes:
                crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                if crop.size == 0:
                    continue
                r = read_plate(crop)
                text = r["plate"] if r["valid"] else (r["raw"] or "...")
                reads.append(((x1 + x2) / 2, (y1 + y2) / 2, text))
            last_ocr = now

        for box in boxes:
            text = nearest_text(box, reads)
            if text:
                draw_label(frame, box, text, (0, 200, 0))
            else:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

        # üst bilgi çubuğu
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now
        cv2.putText(frame, f"FPS: {fps:4.1f}  |  q: cikis  r: hemen oku",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Turk Plaka Okuma - canli", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            last_ocr = 0.0  # sonraki karede OCR'ı zorla

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
