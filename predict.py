"""Eğitilmiş modelle plaka tespiti ve plaka bölgesini kırpma.

Tek görüntü veya klasör alır; her görüntü için:
- Tespit kutuları çizilmiş kopyayı  <out>/annotated/  altına kaydeder.
- Tespit edilen her plaka bölgesini <out>/crops/      altına ayrı görüntü
  olarak kırpar (bu kırpmalar 2. aşamadaki OCR'ın girdisi olacak).

Kullanım:
    python predict.py --source foto.jpg
    python predict.py --source klasor/ --conf 0.4 --out predictions
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Türk plaka tespiti - çıkarım")
    parser.add_argument("--source", type=Path, required=True,
                        help="Görüntü dosyası veya görüntü klasörü")
    parser.add_argument("--weights", type=Path, default=Path("weights/best.pt"))
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Güven eşiği (varsayılan 0.25)")
    parser.add_argument("--out", type=Path, default=Path("predictions"),
                        help="Çıktı klasörü")
    return parser.parse_args()


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        images = sorted(
            p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise SystemExit(f"HATA: {source} klasöründe görüntü bulunamadı.")
        return images
    raise SystemExit(f"HATA: {source} bulunamadı.")


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise SystemExit(
            f"HATA: {args.weights} bulunamadı. Önce modeli eğitin "
            "(train.ipynb veya train.py) ve best.pt'yi weights/ altına koyun."
        )

    images = collect_images(args.source)
    annotated_dir = args.out / "annotated"
    crops_dir = args.out / "crops"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    total_detections = 0

    for image_path in images:
        result = model.predict(source=str(image_path), conf=args.conf, verbose=False)[0]

        annotated = result.plot()  # kutular çizilmiş BGR görüntü
        cv2.imwrite(str(annotated_dir / image_path.name), annotated)

        original = cv2.imread(str(image_path))
        boxes = result.boxes
        for i, xyxy in enumerate(boxes.xyxy.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = xyxy
            crop = original[max(y1, 0):y2, max(x1, 0):x2]
            if crop.size == 0:
                continue
            crop_name = f"{image_path.stem}_plate{i}{image_path.suffix}"
            cv2.imwrite(str(crops_dir / crop_name), crop)
            total_detections += 1

        print(f"{image_path.name}: {len(boxes)} plaka tespit edildi")

    print(f"\nToplam {len(images)} görüntü işlendi, {total_detections} plaka kırpıldı.")
    print(f"Kutulu görüntüler: {annotated_dir}")
    print(f"Kırpılmış plakalar (OCR girdisi): {crops_dir}")


if __name__ == "__main__":
    main()
