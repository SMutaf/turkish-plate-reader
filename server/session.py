"""Oy toplama (saf) ve camera_id başına oturum durumu.

Bu modül ağır bağımlılık içermez (ultralytics/easyocr yok) — yalnızca
shared.plate_format. Böylece `vote()` ve `SessionStore` model/GPU/kamera
olmadan test edilebilir.

Not: Ultralytics ByteTrack örneklerinin camera_id başına izolasyonu burada
DEĞİL, model yükleme katmanında (server/inference.py) yapılır; bu modül yalnızca
okuma/oy verisini tutar.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from shared.plate_format import format_plate


@dataclass
class Reading:
    """Tek bir karede bir track için üretilmiş ham OCR okuması."""

    raw: str          # OCR'ın temizlenmiş ham çıktısı ([0-9A-Z])
    text: str         # format_plate() sonucu (geçersizse en iyi tahmin)
    valid: bool       # format_plate() geçerli plaka üretti mi
    ocr_conf: float   # OCR güveni (0..1)
    plate_conf: float # plaka tespit güveni (0..1)
    frame_idx: int    # okumanın geldiği kare sırası (yenilik ölçütü)


@dataclass
class VoteResult:
    """vote() çıktısı."""

    text: str      # kazanan plaka metni
    raw: str       # kazanan sonucu üreten ham metin
    valid: bool    # kazanan geçerli bir plaka mı (ana yol) yoksa yedek yol mu
    votes: int     # kazananı DESTEKLEYEN okuma sayısı (schemas.votes ile aynı)
    weight: float  # kazananın toplam ağırlığı = sum(ocr_conf * plate_conf)
    margin: float  # kazanan ağırlığı - ikinci ağırlığı; tek aday varsa = weight


def vote(readings: List[Reading]) -> Optional[VoteResult]:
    """Bir track'in okumalarından tek bir sonuç üretir. SAF fonksiyon.

    Boş liste -> None (istisna fırlatmaz).

    ANA YOL (en az bir valid okuma varsa):
      valid okumaları text'e göre grupla, her grubun ağırlığı
      sum(ocr_conf * plate_conf); kazananı DETERMİNİSTİK beraberlik sırasıyla seç:
        a) daha yüksek toplam ağırlık
        b) eşitse daha çok oy sayısı
        c) yine eşitse daha büyük frame_idx (daha yeni)
        d) yine eşitse alfabetik olarak küçük metin

    YEDEK YOL (hiç valid okuma yok):
      ham metinleri uzunluğa göre grupla; en çok okumanın olduğu uzunluk grubunu
      seç (eşitlikte: daha yüksek toplam ocr_conf, sonra daha kısa uzunluk).
      O grupta 2'den az okuma varsa oylama yapma, en yüksek ocr_conf'lu tek
      okumayı döndür. 2+ ise SADECE o grup içinde karakter-pozisyonu çoğunluk
      oyu yap (pozisyon başına en sık karakter; eşitlikte alfabetik küçük).
      Sonucu format_plate()'ten geçir, valid=False işaretle. Farklı uzunluktaki
      metinler hizalanmaya çalışılmaz.
    """
    if not readings:
        return None

    valid = [r for r in readings if r.valid]
    if valid:
        return _vote_valid(valid)
    return _vote_fallback(readings)


def _vote_valid(valid: List[Reading]) -> VoteResult:
    groups: Dict[str, List[Reading]] = {}
    for r in valid:
        groups.setdefault(r.text, []).append(r)

    candidates = []
    for text, rs in groups.items():
        weight = sum(r.ocr_conf * r.plate_conf for r in rs)
        votes = len(rs)
        max_frame = max(r.frame_idx for r in rs)
        # temsili ham metin: en yüksek ağırlıklı üye (deterministik)
        rep = sorted(rs, key=lambda r: (-(r.ocr_conf * r.plate_conf),
                                        -r.frame_idx, r.raw))[0]
        candidates.append((text, weight, votes, max_frame, rep.raw))

    # Deterministik sıralama; max()'ın varsayılan davranışına GÜVENİLMEZ.
    candidates.sort(key=lambda c: (-c[1], -c[2], -c[3], c[0]))
    win = candidates[0]
    margin = win[1] - candidates[1][1] if len(candidates) > 1 else win[1]
    return VoteResult(text=win[0], raw=win[4], valid=True,
                      votes=win[2], weight=win[1], margin=margin)


def _vote_fallback(readings: List[Reading]) -> VoteResult:
    by_len: Dict[int, List[Reading]] = {}
    for r in readings:
        by_len.setdefault(len(r.raw), []).append(r)

    # en çok okuma; eşitlikte yüksek toplam ocr_conf; sonra kısa uzunluk
    length, group = sorted(
        by_len.items(),
        key=lambda kv: (-len(kv[1]), -sum(x.ocr_conf for x in kv[1]), kv[0]),
    )[0]

    if len(group) < 2:
        best = sorted(group, key=lambda r: (-r.ocr_conf, -r.frame_idx, r.raw))[0]
        voted_raw = best.raw
        weight = best.ocr_conf * best.plate_conf
        votes = 1
    else:
        voted_chars = []
        for i in range(length):
            counts: Dict[str, int] = {}
            for r in group:
                counts[r.raw[i]] = counts.get(r.raw[i], 0) + 1
            # en sık; eşitlikte alfabetik küçük
            ch = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            voted_chars.append(ch)
        voted_raw = "".join(voted_chars)
        weight = sum(r.ocr_conf * r.plate_conf for r in group)
        votes = len(group)

    text = format_plate(voted_raw) or voted_raw
    return VoteResult(text=text, raw=voted_raw, valid=False,
                      votes=votes, weight=weight, margin=weight)


class SessionStore:
    """camera_id -> track_id -> son N okuma (deque). Zaman içeriden ÇAĞRILMAZ."""

    def __init__(self, vote_window: int):
        self.vote_window = vote_window
        self._readings: Dict[str, Dict[int, Deque[Reading]]] = defaultdict(dict)
        self._last_seen: Dict[str, Dict[int, float]] = defaultdict(dict)

    def add(self, camera_id: str, track_id: int, reading: Reading, ts: float) -> None:
        tracks = self._readings[camera_id]
        if track_id not in tracks:
            tracks[track_id] = deque(maxlen=self.vote_window)
        tracks[track_id].append(reading)
        self._last_seen[camera_id][track_id] = ts

    def get_readings(self, camera_id: str, track_id: int) -> List[Reading]:
        return list(self._readings.get(camera_id, {}).get(track_id, ()))

    def touch(self, camera_id: str, track_id: int, ts: float) -> None:
        """Var olan bir track'in son görülme zamanını günceller (yeni okuma
        eklemeden). Bu karede görülen ama OCR yapılmayan tracklerin erken
        silinmesini önler."""
        if track_id in self._last_seen.get(camera_id, {}):
            self._last_seen[camera_id][track_id] = ts

    def evict_stale(self, now_ts: float, max_age_sec: float) -> int:
        """now_ts - son_görülme > max_age_sec olan track'leri siler.

        Zamanı DIŞARIDAN alır; içeride time.time() çağırmaz. now_ts ve add()'e
        verilen ts AYNI birimde olmalı (eşik de o birimde). Silinen track sayısını
        döndürür.
        """
        removed = 0
        for camera_id in list(self._readings):
            tracks = self._readings[camera_id]
            for track_id in list(tracks):
                if now_ts - self._last_seen[camera_id][track_id] > max_age_sec:
                    del tracks[track_id]
                    del self._last_seen[camera_id][track_id]
                    removed += 1
            if not tracks:
                del self._readings[camera_id]
                self._last_seen.pop(camera_id, None)
        return removed
