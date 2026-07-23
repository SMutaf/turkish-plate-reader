"""Aşama 2 — OCR: kırpılmış plaka görüntüsünden metni okur.

Boru hattı:
    kırpma -> ön işleme (gri/CLAHE/büyütme) -> EasyOCR -> Türk plaka
    format doğrulaması/düzeltmesi -> "34 ABC 123"

Türk plaka format doğrulaması shared/plate_format.py'ye taşındı (mantık aynı).

Tek başına kullanım (bir klasördeki kırpılmış plakaları oku):
    python ocr.py --source predictions/crops
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

from shared.plate_format import ALLOWLIST, format_plate

_reader = None


def get_reader():
    """EasyOCR okuyucusunu tek sefer başlatır (ilk çağrıda model iner)."""
    global _reader
    if _reader is None:
        import easyocr  # ağır import; yalnızca gerekince yükle
        import torch
        _reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    return _reader


def preprocess(crop: np.ndarray) -> np.ndarray:
    """Plaka kırpmasını OCR için hazırlar: büyütme, gri, gürültü azaltma,
    kontrast (CLAHE) ve keskinleştirme."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    target_h = 96
    if h < target_h:  # küçük kırpmaları belirgin şekilde büyüt
        scale = target_h / max(h, 1)
        gray = cv2.resize(gray, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)  # kenarları koruyarak gürültü azalt
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)  # unsharp mask ile keskinleştir
    return cv2.addWeighted(gray, 1.5, blur, -0.5, 0)


def read_plate(crop: np.ndarray) -> dict:
    """Bir plaka kırpmasını okur.

    Döndürür: {"raw": ham okuma, "plate": doğrulanmış metin veya None,
               "valid": bool}
    """
    reader = get_reader()
    img = preprocess(crop)
    parts = reader.readtext(img, allowlist=ALLOWLIST, detail=0, paragraph=False)
    raw = re.sub(r"[^0-9A-Z]", "", "".join(parts).upper())
    plate = format_plate(raw)
    return {"raw": raw, "plate": plate, "valid": plate is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Kırpılmış plakalardan metin okuma")
    parser.add_argument("--source", type=Path, required=True,
                        help="Kırpılmış plaka görüntüsü veya klasörü")
    args = parser.parse_args()

    if args.source.is_dir():
        images = sorted(p for p in args.source.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    else:
        images = [args.source]
    if not images:
        raise SystemExit(f"HATA: {args.source} içinde görüntü yok.")

    for path in images:
        crop = cv2.imread(str(path))
        if crop is None:
            continue
        result = read_plate(crop)
        status = result["plate"] if result["valid"] else f"(geçersiz: {result['raw']})"
        print(f"{path.name}: {status}")


if __name__ == "__main__":
    main()
