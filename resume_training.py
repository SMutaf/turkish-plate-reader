"""Kesilen eğitimi en güncel kontrol noktasından 50. epoch'a kadar devam ettirir,
sonra değerlendirir. Kendi terminalinizde çalıştırmak için:

    python resume_training.py

Not: Bu, kesilen bir eğitimi kurtarmak içindir. Sıfırdan eğitim için train.py kullanın.
"""
import random
import shutil
from multiprocessing import freeze_support
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path("datasets/turkish-plates/data.yaml")


def latest_checkpoint() -> Path:
    ckpts = sorted(Path("runs").glob("**/plate-detect*/weights/last.pt"),
                   key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise SystemExit("last.pt bulunamadı — önce train.py ile eğitime başlayın.")
    return ckpts[-1]


def main() -> None:
    ckpt = latest_checkpoint()
    print("Devam edilen kontrol noktası:", ckpt)
    model = YOLO(str(ckpt))
    model.train(resume=True)

    best = Path(model.trainer.best)
    Path("weights").mkdir(exist_ok=True)
    shutil.copy2(best, "weights/best.pt")
    print("EN IYI AGIRLIK:", best, "-> weights/best.pt")

    m = YOLO("weights/best.pt")
    metrics = m.val(data=str(DATA_YAML), split="val")
    rows = {
        "mAP@0.5": metrics.box.map50,
        "mAP@0.5:0.95": metrics.box.map,
        "Precision": metrics.box.mp,
        "Recall": metrics.box.mr,
    }
    Path("results").mkdir(exist_ok=True)
    lines = ["| Metrik | Değer |", "|---|---|"] + [f"| {k} | {v:.4f} |" for k, v in rows.items()]
    Path("results/metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("METRIKLER:", {k: round(v, 4) for k, v in rows.items()})

    val_dir = DATA_YAML.parent / "valid" / "images"
    imgs = sorted(val_dir.glob("*.jpg")) + sorted(val_dir.glob("*.png"))
    sample = random.sample(imgs, min(8, len(imgs)))
    m.predict(source=[str(p) for p in sample], save=True,
              project="results", name="val_predictions", exist_ok=True)
    print("TAMAMLANDI")


if __name__ == "__main__":
    freeze_support()
    main()
