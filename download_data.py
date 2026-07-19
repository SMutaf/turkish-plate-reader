"""Roboflow'dan Türk plaka veri setini YOLOv8 formatında indirir.

Kullanım:
    export ROBOFLOW_API_KEY="anahtariniz"   # veya .env dosyasına yazın
    python download_data.py

Veri seti: license-plates-of-vehicles-in-turkey (CC BY 4.0)
https://universe.roboflow.com/kemalkilicaslan-gzpvq/license-plates-of-vehicles-in-turkey-s3tbj
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

WORKSPACE = "kemalkilicaslan-gzpvq"
PROJECT = "license-plates-of-vehicles-in-turkey-s3tbj"
DATASET_DIR = Path("datasets")


def main() -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit(
            "HATA: ROBOFLOW_API_KEY ortam değişkeni tanımlı değil.\n"
            "Anahtarınızı https://app.roboflow.com/settings/api adresinden alıp\n"
            "  export ROBOFLOW_API_KEY=\"...\"   (Linux/macOS)\n"
            "  $env:ROBOFLOW_API_KEY=\"...\"     (Windows PowerShell)\n"
            "şeklinde tanımlayın ya da proje kökünde bir .env dosyasına yazın."
        )

    from roboflow import Roboflow

    DATASET_DIR.mkdir(exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(get_latest_version(project))
    dataset = version.download(
        model_format="yolov8", location=str(DATASET_DIR / "turkish-plates")
    )

    fix_data_yaml(Path(dataset.location))

    print(f"\nVeri seti indirildi: {dataset.location}")
    print(f"data.yaml yolu: {Path(dataset.location) / 'data.yaml'}")


def fix_data_yaml(dataset_dir: Path) -> None:
    """data.yaml'daki train/val/test yollarını indirilen konuma göre mutlaklaştırır.

    Roboflow export'u bazen `../train/images` gibi göreli yollar içerir; bunlar
    repo kökünden çalıştırıldığında yanlış yere çözülür. Bölünmenin kendisine
    dokunulmaz, yalnızca yollar düzeltilir.
    """
    import yaml

    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        return

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    changed = False
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        split_dir = dataset_dir / folder / "images"
        if split in data and split_dir.exists():
            data[split] = str(split_dir.resolve())
            changed = True
    if changed:
        yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def get_latest_version(project) -> int:
    versions = project.versions()
    if not versions:
        sys.exit("HATA: Projede yayınlanmış bir veri seti sürümü bulunamadı.")
    return max(int(v.version.split("/")[-1]) for v in versions)


if __name__ == "__main__":
    main()
