"""YOLOv8n ile Türk plaka tespit modelini eğitir ve değerlendirir.

Kullanım (önce `python download_data.py` ile veriyi indirin):
    python train.py
    python train.py --epochs 50 --imgsz 640 --batch 16

Eğitim sonunda:
- En iyi ağırlık weights/best.pt olarak kopyalanır.
- Val seti metrikleri (mAP@0.5, mAP@0.5:0.95, precision, recall) yazdırılır
  ve results/metrics.md dosyasına kaydedilir.
- Birkaç val görüntüsü üzerinde tahmin kutuları çizilip results/ altına kaydedilir.
"""

import argparse
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path("datasets/turkish-plates/data.yaml")
WEIGHTS_DIR = Path("weights")
RESULTS_DIR = Path("results")
NUM_SAMPLE_PREDICTIONS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Türk plaka tespiti - YOLOv8 eğitimi")
    parser.add_argument("--data", type=Path, default=DATA_YAML, help="data.yaml yolu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16,
                        help="GPU belleği yetmezse düşürün (ör. 8)")
    parser.add_argument("--model", default="yolov8n.pt", help="Başlangıç ağırlığı")
    return parser.parse_args()


def train(args: argparse.Namespace) -> YOLO:
    if not args.data.exists():
        raise SystemExit(
            f"HATA: {args.data} bulunamadı. Önce `python download_data.py` çalıştırın."
        )

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs",
        name="plate-detect",
    )

    best = Path(model.trainer.best)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    shutil.copy2(best, WEIGHTS_DIR / "best.pt")
    print(f"\nEn iyi ağırlık kopyalandı: {WEIGHTS_DIR / 'best.pt'}")
    return YOLO(WEIGHTS_DIR / "best.pt")


def evaluate(model: YOLO, data_yaml: Path) -> None:
    """Val seti metriklerini raporlar ve results/metrics.md'ye yazar."""
    metrics = model.val(data=str(data_yaml), split="val")

    rows = {
        "mAP@0.5": metrics.box.map50,
        "mAP@0.5:0.95": metrics.box.map,
        "Precision": metrics.box.mp,
        "Recall": metrics.box.mr,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    lines = ["| Metrik | Değer |", "|---|---|"]
    print("\n=== Val seti metrikleri ===")
    for name, value in rows.items():
        print(f"{name:14s}: {value:.4f}")
        lines.append(f"| {name} | {value:.4f} |")
    (RESULTS_DIR / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nMetrik tablosu kaydedildi: {RESULTS_DIR / 'metrics.md'}")
    print("Bu tabloyu README'deki 'Değerlendirme Sonuçları' bölümüne yapıştırın.")


def save_sample_predictions(model: YOLO, data_yaml: Path) -> None:
    """Rastgele birkaç val görüntüsünde tahmin kutularını çizip results/ altına kaydeder."""
    val_images_dir = data_yaml.parent / "valid" / "images"
    if not val_images_dir.exists():
        print(f"Uyarı: {val_images_dir} bulunamadı, örnek tahminler atlandı.")
        return

    images = sorted(val_images_dir.glob("*.jpg")) + sorted(val_images_dir.glob("*.png"))
    if not images:
        print("Uyarı: Val klasöründe görüntü bulunamadı.")
        return

    sample = random.sample(images, min(NUM_SAMPLE_PREDICTIONS, len(images)))
    out_dir = RESULTS_DIR / "val_predictions"
    model.predict(
        source=[str(p) for p in sample],
        save=True,
        project=str(RESULTS_DIR),
        name="val_predictions",
        exist_ok=True,
    )
    print(f"Örnek tahminler kaydedildi: {out_dir}")


def main() -> None:
    args = parse_args()
    model = train(args)
    evaluate(model, args.data)
    save_sample_predictions(model, args.data)


if __name__ == "__main__":
    main()
