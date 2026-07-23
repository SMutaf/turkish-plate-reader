"""Türk plaka format doğrulaması ve OCR karışıklık düzeltmesi.

Bu mantık ocr.py'den TAŞINDI; içeriği aynen korunmuştur. Ağır bağımlılık yok
(yalnızca `re`), böylece CLI (ocr.py), sunucu (server/) ve testler bu modülü
model/GPU olmadan kullanabilir.
"""
from __future__ import annotations

import re

# Türk plakalarında kullanılan harfler (Q, W, X ve Türkçe'ye özgü harfler yok)
PLATE_LETTERS = "ABCDEFGHIJKLMNOPRSTUVYZ"
ALLOWLIST = "0123456789" + PLATE_LETTERS

# Konuma göre OCR karışıklık düzeltmeleri
LETTER_TO_DIGIT = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8",
                   "G": "6", "A": "4", "T": "7", "Q": "0", "D": "0"}
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B",
                   "6": "G", "4": "A", "7": "T"}


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
