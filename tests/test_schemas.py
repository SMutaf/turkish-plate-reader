"""server.schemas testleri (model/kamera gerektirmez)."""
import json

import pytest
from pydantic import ValidationError

from server.schemas import InferResponse

# Elle yazılmış, sözleşmeyi temsil eden örnek cevap
EXAMPLE = {
    "camera_id": "kapi1",
    "frame_ts": 1730000000000,
    "latency_ms": 42,
    "frame_w": 1280,
    "frame_h": 720,
    "vehicles": [
        {
            "track_id": 7,
            "class": "car",
            "box": [10, 20, 110, 120],
            "conf": 0.91,
            "plate": {
                "box": [40, 90, 100, 110],
                "conf": 0.8,
                "raw": "54ZP236",
                "text": "54 ZP 236",
                "valid": True,
                "votes": 3,
            },
        },
        {
            "track_id": 8,
            "class": "truck",
            "box": [200, 50, 400, 300],
            "conf": 0.77,
            "plate": None,
        },
    ],
    # orphan plakalara OCR uygulanmaz -> text daima null
    "orphan_plates": [
        {"box": [500, 500, 560, 520], "conf": 0.4, "text": None}
    ],
}


def test_roundtrip_preserves_field_names():
    """Parse -> serialize alan adlarını birebir korumalı ('class' dahil)."""
    model = InferResponse.model_validate(EXAMPLE)
    dumped = model.model_dump(by_alias=True)
    assert dumped == EXAMPLE


def test_class_alias_serialized_as_class():
    dumped = InferResponse.model_validate(EXAMPLE).model_dump(by_alias=True)
    assert "class" in dumped["vehicles"][0]
    assert "vehicle_class" not in dumped["vehicles"][0]


def test_plate_null_is_valid():
    model = InferResponse.model_validate(EXAMPLE)
    assert model.vehicles[1].plate is None


def test_empty_vehicles_list_is_valid():
    data = json.loads(json.dumps(EXAMPLE))
    data["vehicles"] = []
    data["orphan_plates"] = []
    model = InferResponse.model_validate(data)
    assert model.vehicles == []
    assert model.model_dump(by_alias=True)["vehicles"] == []


def test_track_id_null_is_valid():
    data = json.loads(json.dumps(EXAMPLE))
    data["vehicles"][0]["track_id"] = None   # tracker atayamadı
    model = InferResponse.model_validate(data)
    assert model.vehicles[0].track_id is None
    assert model.model_dump(by_alias=True)["vehicles"][0]["track_id"] is None


def test_conf_out_of_range_rejected():
    data = json.loads(json.dumps(EXAMPLE))
    data["vehicles"][0]["conf"] = 1.5  # 0..1 dışı
    with pytest.raises(ValidationError):
        InferResponse.model_validate(data)


def test_invalid_vehicle_class_rejected():
    data = json.loads(json.dumps(EXAMPLE))
    data["vehicles"][0]["class"] = "bicycle"  # izinli küme dışı
    with pytest.raises(ValidationError):
        InferResponse.model_validate(data)


def test_box_must_have_four_ints():
    data = json.loads(json.dumps(EXAMPLE))
    data["vehicles"][0]["box"] = [1, 2, 3]  # eksik
    with pytest.raises(ValidationError):
        InferResponse.model_validate(data)
