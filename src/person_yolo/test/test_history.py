from person_yolo.history import History


def filled(n=50, dt=0.05, t0=100.0):
    h = History(keep_sec=3.0)
    for i in range(n):
        h.add(t0 + i * dt, i)
    return h


def test_nearest_sample():
    h = filled()                              # 100.00, 100.05, ... 102.45
    assert h.at(100.51) == (10, 100.5)
    assert h.at(100.53)[0] == 11              # más cerca de 100.55
    assert h.at(100.0)[0] == 0


def test_none_outside_history_or_empty():
    assert History().at(5.0) is None
    h = filled()
    assert h.at(103.0) is None                # 0.55 s después de la última
    assert h.at(99.0) is None                 # antes de la primera
    assert h.at(102.6, max_gap=0.2)[0] == 49  # cerca del final: la última


def test_keeps_only_keep_sec():
    h = filled(n=200)                         # 10 s de datos
    assert len(h) <= 61                       # 3 s a 20 Hz (+ borde)
    assert h.at(100.5) is None


def test_stamp_going_back_restarts():
    h = filled()
    h.add(10.0, 'nuevo')                      # rosbag que vuelve a empezar
    assert len(h) == 1 and h.at(10.0) == ('nuevo', 10.0)
