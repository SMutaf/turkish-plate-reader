"""Uçtan uca boru hattı: tespit (Aşama 1) + OCR (Aşama 2).

Bir görüntü/klasör alır; her araç için plakayı YOLOv8 ile bulur, bölgeyi
kırpar, OCR ile metni okur ve okunan plakayı görüntü üzerine yazıp kaydeder.

Kullanım:
    python pipeline.py --source foto.jpg
    python pipeline.py --source test_fotolarim/ --conf 0.25 --out pipeline_out
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from ocr import read_plate

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Türk plaka tespiti + OCR (uçtan uca)")
    parser.add_argument("--source", type=Path, required=True,
                        help="Görüntü dosyası veya klasörü")
    parser.add_argument("--weights", type=Path, default=Path("weights/best.pt"))
    parser.add_argument("--conf", type=float, default=0.25, help="Tespit güven eşiği")
    parser.add_argument("--out", type=Path, default=Path("pipeline_out"))
    return parser.parse_args()


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        images = sorted(p for p in source.iterdir()
                        if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise SystemExit(f"HATA: {source} içinde görüntü yok.")
        return images
    raise SystemExit(f"HATA: {source} bulunamadı.")


def draw_label(img, box, text: str) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw, y1), (0, 200, 0), -1)
    cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise SystemExit(f"HATA: {args.weights} bulunamadı. Önce Aşama 1 modelini eğitin.")

    images = collect_images(args.source)
    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    total_read = 0
    for image_path in images:
        result = model.predict(source=str(image_path), conf=args.conf, verbose=False)[0]
        img = cv2.imread(str(image_path))
        plates = []
        for xyxy in result.boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = xyxy
            crop = img[max(y1, 0):y2, max(x1, 0):x2]
            if crop.size == 0:
                continue
            ocr = read_plate(crop)
            text = ocr["plate"] if ocr["valid"] else f"? {ocr['raw']}"
            plates.append(text)
            if ocr["valid"]:
                total_read += 1
            draw_label(img, (x1, y1, x2, y2), text)

        cv2.imwrite(str(args.out / image_path.name), img)
        print(f"{image_path.name}: {plates if plates else 'plaka yok'}")

    print(f"\n{len(images)} görüntü işlendi, {total_read} plaka okundu.")
    print(f"Sonuçlar: {args.out}")


if __name__ == "__main__":
    main()
