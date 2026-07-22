"""Aşama 2 — OCR: kırpılmış plaka görüntüsünden metni okur.

Boru hattı:
    kırpma -> ön işleme (gri/CLAHE/büyütme) -> EasyOCR -> Türk plaka
    format doğrulaması/düzeltmesi -> "34 ABC 123"

Tek başına kullanım (bir klasördeki kırpılmış plakaları oku):
    python ocr.py --source predictions/crops
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

# Türk plakalarında kullanılan harfler (Q, W, X ve Türkçe'ye özgü harfler yok)
PLATE_LETTERS = "ABCDEFGHIJKLMNOPRSTUVYZ"
ALLOWLIST = "0123456789" + PLATE_LETTERS

# Konuma göre OCR karışıklık düzeltmeleri
LETTER_TO_DIGIT = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8",
                   "G": "6", "A": "4", "T": "7", "Q": "0", "D": "0"}
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B",
                   "6": "G", "4": "A", "7": "T"}

# Türk plaka deseni: 2 rakam (il 01-81) + 1-3 harf + 2-4 rakam
PLATE_RE = re.compile(r"^([0-9]{2})([A-Z]{1,3})([0-9]{2,4})$")

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


def _coerce(segment: str, to_digit: bool) -> tuple[str, int] | None:
    """Bir segmenti tümüyle rakama (to_digit) ya da harfe zorlar.

    Döndürür: (dönüştürülmüş segment, gereken değişiklik sayısı) veya
    dönüştürülemiyorsa None. Değişiklik sayısı en az zorlamalı bölünmeyi
    seçmek için kullanılır.
    """
    out, changes = [], 0
    for c in segment:
        if to_digit:
            if c.isdigit():
                out.append(c)
            elif c in LETTER_TO_DIGIT:
                out.append(LETTER_TO_DIGIT[c]); changes += 1
            else:
                return None
        else:
            if c in PLATE_LETTERS:
                out.append(c)
            elif c in DIGIT_TO_LETTER:
                out.append(DIGIT_TO_LETTER[c]); changes += 1
            else:
                return None
    return "".join(out), changes


def format_plate(raw: str) -> str | None:
    """Ham OCR çıktısını Türk plaka formatına oturtmaya çalışır.

    Harf/rakam bölünmesini, **en az karakter zorlaması** gerektirecek şekilde
    seçer (ör. rakam zaten rakamsa harfe çevirmez). Başarılıysa 'IL HARF RAKAM'
    döndürür (ör. '34 ABC 123'), aksi halde None.
    """
    s = re.sub(r"[^0-9A-Z]", "", raw.upper())
    if not (5 <= len(s) <= 9):
        return None

    prov_coerced = _coerce(s[:2], to_digit=True)
    if prov_coerced is None:
        return None
    prov, prov_changes = prov_coerced
    if not (1 <= int(prov) <= 81):
        return None

    body = s[2:]
    best, best_changes = None, None
    for n_letters in (1, 2, 3):
        n_digits = len(body) - n_letters
        if not (2 <= n_digits <= 4):
            continue
        letters = _coerce(body[:n_letters], to_digit=False)
        digits = _coerce(body[n_letters:], to_digit=True)
        if letters is None or digits is None:
            continue
        changes = prov_changes + letters[1] + digits[1]
        if best_changes is None or changes < best_changes:
            best_changes = changes
            best = f"{prov} {letters[0]} {digits[0]}"
    return best


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
