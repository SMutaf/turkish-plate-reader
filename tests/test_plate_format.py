"""shared.plate_format.format_plate testleri (model/kamera gerektirmez)."""
import pytest

from shared.plate_format import format_plate

VALID_CASES = [
    # (ham OCR, beklenen plaka)
    ("34ABC123", "34 ABC 123"),   # temel geçerli
    ("06A12", "06 A 12"),         # 1 harf, 2 rakam
    ("06BCD45", "06 BCD 45"),     # 3 harf
    ("81YZ9999", "81 YZ 9999"),   # en yüksek il (81), 4 rakam
    ("54ZP236", "54 ZP 236"),     # min-zorlama: '2' harfe çevrilmemeli (ZPZ değil ZP)
    ("O6ABC12", "06 ABC 12"),     # il kodunda O -> 0 karışıklığı
    ("34ABC1I", "34 ABC 11"),     # rakam bölgesinde I -> 1 karışıklığı
    ("340BC12", "34 OBC 12"),     # harf bölgesinde 0 -> O karışıklığı
    ("54ZPZ36", "54 ZPZ 36"),     # gerçekten 3 harf (daha iyi bölünme yok)
]

INVALID_CASES = [
    "00ABC12",     # il kodu 00 geçersiz
    "82ABC12",     # il kodu 82 geçersiz (>81)
    "99ABC12",     # il kodu 99 geçersiz
    "34AB",        # çok kısa (temizlenince len 4)
    "34ABCDE123",  # çok uzun (len 10)
    "34ABC",       # geçerli rakam kuyruğu yok
]


@pytest.mark.parametrize("raw,expected", VALID_CASES)
def test_format_plate_valid(raw, expected):
    assert format_plate(raw) == expected


@pytest.mark.parametrize("raw", INVALID_CASES)
def test_format_plate_invalid(raw):
    assert format_plate(raw) is None


def test_lowercase_and_noise_cleaned():
    # küçük harf ve gürültü karakterleri temizlenip yine de okunmalı
    assert format_plate("34-abc-123") == "34 ABC 123"


def test_min_coercion_prefers_fewer_changes():
    # '54ZP236' iki geçerli bölünme adayına sahip; en az zorlamalı seçilmeli
    assert format_plate("54ZP236") == "54 ZP 236"
