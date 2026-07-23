"""server.inference.match_plates_to_vehicles testleri (modelsiz)."""
from server.inference import PlateDet, match_plates_to_vehicles


def P(box, conf=0.9):
    return PlateDet(box=list(box), conf=conf)


def test_plate_center_inside_vehicle():
    veh = [[0, 0, 100, 100]]
    res = match_plates_to_vehicles(veh, [P([40, 40, 60, 60])])
    assert 0 in res.vehicle_plate
    assert res.orphan_plates == []
    assert res.secondary_plates == []


def test_plate_center_outside_all_is_orphan():
    veh = [[0, 0, 100, 100]]
    res = match_plates_to_vehicles(veh, [P([200, 200, 260, 220])])
    assert res.vehicle_plate == {}
    assert len(res.orphan_plates) == 1


def test_partial_overlap_center_outside_is_orphan():
    # plaka kutusu araçla örtüşüyor ama MERKEZİ (120,120) araç dışında
    veh = [[0, 0, 100, 100]]
    res = match_plates_to_vehicles(veh, [P([90, 90, 150, 150])])
    assert res.vehicle_plate == {}
    assert len(res.orphan_plates) == 1


def test_nested_vehicles_smallest_area_wins():
    # 0: büyük, 1: küçük (iç içe). Merkez (100,100) ikisinde de -> küçük kazanır
    veh = [[0, 0, 200, 200], [50, 50, 150, 150]]
    res = match_plates_to_vehicles(veh, [P([95, 95, 105, 105])])
    assert 1 in res.vehicle_plate
    assert 0 not in res.vehicle_plate


def test_two_plates_one_vehicle_highest_conf_primary():
    veh = [[0, 0, 100, 100]]
    plates = [P([20, 20, 40, 40], conf=0.6), P([60, 60, 80, 80], conf=0.9)]
    res = match_plates_to_vehicles(veh, plates)
    assert res.vehicle_plate[0].conf == 0.9         # yüksek conf birincil
    assert len(res.secondary_plates) == 1
    assert res.secondary_plates[0].conf == 0.6      # diğeri secondary (orphan DEĞİL)
    assert res.orphan_plates == []


def test_no_vehicles_all_orphan():
    res = match_plates_to_vehicles([], [P([10, 10, 20, 20])])
    assert res.vehicle_plate == {}
    assert len(res.orphan_plates) == 1


def test_no_plates_empty_result():
    res = match_plates_to_vehicles([[0, 0, 100, 100]], [])
    assert res.vehicle_plate == {}
    assert res.orphan_plates == []
    assert res.secondary_plates == []
