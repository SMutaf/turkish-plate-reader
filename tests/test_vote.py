"""server.session.vote() ve SessionStore testleri (model/kamera gerektirmez)."""
from server.session import Reading, SessionStore, vote


def mk(text, ocr, plate, frame, valid=True, raw=None):
    """Kısa Reading kurucu. raw verilmezse text'ten boşluklar atılarak üretilir."""
    if raw is None:
        raw = text.replace(" ", "")
    return Reading(raw=raw, text=text, valid=valid,
                   ocr_conf=ocr, plate_conf=plate, frame_idx=frame)


# ---- boş / tek okuma ----

def test_empty_returns_none():
    assert vote([]) is None


def test_single_valid_reading():
    r = mk("54 ZP 236", 0.8, 0.9, frame=1)
    res = vote([r])
    assert res.text == "54 ZP 236"
    assert res.valid is True
    assert res.votes == 1
    assert res.weight == 0.8 * 0.9
    assert res.margin == 0.8 * 0.9  # tek aday -> margin = weight


# ---- net çoğunluk ----

def test_clear_majority():
    readings = [
        mk("34 ABC 12", 0.7, 0.9, 1),
        mk("34 ABC 12", 0.8, 0.9, 2),
        mk("34 ABC 12", 0.6, 0.9, 3),
        mk("34 XYZ 99", 0.5, 0.5, 4),  # not: X/W/Q TR harfi değil ama vote text'e bakar
    ]
    res = vote(readings)
    assert res.text == "34 ABC 12"
    assert res.valid is True
    assert res.votes == 3


# ---- ağırlık, oy sayısını yener ----

def test_weight_beats_vote_count():
    # 1 yüksek güvenli (ağırlık 0.9) vs 3 düşük güvenli (toplam 0.3)
    readings = [
        mk("06 A 11", 0.9, 1.0, 1),          # weight 0.9, votes 1
        mk("06 B 22", 0.1, 1.0, 2),
        mk("06 B 22", 0.1, 1.0, 3),
        mk("06 B 22", 0.1, 1.0, 4),          # weight 0.3, votes 3
    ]
    res = vote(readings)
    assert res.text == "06 A 11"   # ağırlık kazanır
    assert res.votes == 1


# ---- dört beraberlik kuralı ayrı ayrı ----

def test_tiebreak_a_higher_weight():
    res = vote([mk("06 A 11", 0.9, 1.0, 1), mk("06 B 22", 0.5, 1.0, 1)])
    assert res.text == "06 A 11"


def test_tiebreak_b_more_votes_when_weight_equal():
    # X: 1 okuma weight 1.0 ; Y: 2 okuma weight 0.5+0.5=1.0 -> eşit ağırlık, Y çok oy
    readings = [
        mk("06 X 11", 1.0, 1.0, 1),
        mk("06 Y 22", 0.5, 1.0, 1),
        mk("06 Y 22", 0.5, 1.0, 1),
    ]
    res = vote(readings)
    assert res.text == "06 Y 22"
    assert res.votes == 2


def test_tiebreak_c_higher_frame_when_weight_and_votes_equal():
    readings = [
        mk("06 X 11", 1.0, 1.0, frame=5),
        mk("06 Y 22", 1.0, 1.0, frame=9),
    ]
    res = vote(readings)
    assert res.text == "06 Y 22"   # daha büyük frame_idx


def test_tiebreak_d_alphabetical_when_all_equal():
    readings = [
        mk("06 B 11", 1.0, 1.0, frame=5),
        mk("06 A 11", 1.0, 1.0, frame=5),
    ]
    res = vote(readings)
    assert res.text == "06 A 11"   # alfabetik küçük


# ---- yedek yol: hiç valid okuma yok ----

def test_fallback_different_lengths_pick_modal():
    # iki uzunluk-6, bir uzunluk-7; uzunluk-6 grubu seçilmeli
    readings = [
        mk("", 0.5, 0.8, 1, valid=False, raw="34AB12"),
        mk("", 0.6, 0.8, 2, valid=False, raw="34AB12"),
        mk("", 0.9, 0.8, 3, valid=False, raw="34AB123"),  # aykırı uzunluk, yok sayılır
    ]
    res = vote(readings)
    assert res.valid is False
    assert res.raw == "34AB12"          # uzunluk-6 grubunun oyu
    assert res.votes == 2               # sadece o grubun okumaları
    assert res.text == "34 AB 12"       # format_plate en iyi tahmin (ama valid False)


def test_fallback_single_reading_returns_highest_conf():
    readings = [
        mk("", 0.3, 0.8, 1, valid=False, raw="06ABC12"),
        mk("", 0.9, 0.8, 2, valid=False, raw="34XY99"),  # farklı uzunluk, tek başına grup
    ]
    res = vote(readings)
    assert res.valid is False
    # her uzunluk grubu 1 okuma; en çok okuma eşit -> yüksek toplam ocr_conf kazanır
    assert res.raw == "34XY99"
    assert res.votes == 1


# ---- SessionStore ----

def test_deque_overflow_drops_oldest():
    store = SessionStore(vote_window=3)
    for i in range(4):
        store.add("cam1", 7, mk("06 A 11", 0.5, 0.5, frame=i), ts=i)
    readings = store.get_readings("cam1", 7)
    assert len(readings) == 3
    assert readings[0].frame_idx == 1   # en eski (frame 0) düştü
    assert readings[-1].frame_idx == 3


def test_evict_stale_uses_param_time():
    store = SessionStore(vote_window=5)
    store.add("cam1", 1, mk("06 A 11", 0.5, 0.5, 0), ts=100.0)
    # eşik içinde: silinmez
    assert store.evict_stale(now_ts=105.0, max_age_sec=10.0) == 0
    assert len(store.get_readings("cam1", 1)) == 1
    # eşik aşıldı: silinir
    assert store.evict_stale(now_ts=120.0, max_age_sec=10.0) == 1
    assert store.get_readings("cam1", 1) == []


def test_touch_refreshes_last_seen():
    store = SessionStore(vote_window=5)
    store.add("cam1", 1, mk("06 A 11", 0.5, 0.5, 0), ts=100.0)
    store.touch("cam1", 1, ts=110.0)          # görüldü, tazelendi
    assert store.evict_stale(now_ts=115.0, max_age_sec=10.0) == 0  # 115-110=5 <=10
    assert store.evict_stale(now_ts=125.0, max_age_sec=10.0) == 1  # 125-110=15 >10


def test_touch_ignores_unknown_track():
    store = SessionStore(vote_window=5)
    store.touch("cam1", 999, ts=100.0)        # yok, sessizce yok sayılır
    assert store.get_readings("cam1", 999) == []


def test_camera_isolation_same_track_id():
    store = SessionStore(vote_window=5)
    store.add("camA", 5, mk("34 AAA 11", 0.5, 0.5, 1), ts=1)
    store.add("camB", 5, mk("06 BBB 22", 0.5, 0.5, 1), ts=1)
    a = store.get_readings("camA", 5)
    b = store.get_readings("camB", 5)
    assert len(a) == 1 and len(b) == 1
    assert a[0].text == "34 AAA 11"
    assert b[0].text == "06 BBB 22"   # karışmadı
