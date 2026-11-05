import numpy as np

from person_yolo.tracker import PersonCounter


def _box(cx, cy, w=40, h=80, score=0.9):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score, 0]


def _run(counter, frames, dt=0.1):
    out = []
    for i, boxes in enumerate(frames):
        out.append(counter.update(np.array(boxes).reshape(-1, 6), i * dt))
    return out


def test_same_person_counted_once():
    c = PersonCounter(min_hits=3)
    ids = _run(c, [[_box(100 + 5 * i, 200)] for i in range(30)])
    assert c.total == 1
    assert ids[0][0] == 0 and ids[1][0] == 0     # sin confirmar todavía
    assert all(f[0] == 1 for f in ids[2:])


def test_two_people_two_ids():
    c = PersonCounter(min_hits=3)
    ids = _run(c, [[_box(100, 200), _box(500, 200)] for _ in range(10)])
    assert c.total == 2
    assert set(ids[-1]) == {1, 2}


def test_single_frame_false_positive_not_counted():
    c = PersonCounter(min_hits=3)
    _run(c, [[_box(300, 300)], [], [], [], []])
    assert c.total == 0


def test_short_gap_keeps_id():
    # Frames sin detección (p.ej. sin tiles) más cortos que max_age_sec.
    c = PersonCounter(min_hits=2, max_age_sec=1.0)
    frames = [[_box(100, 100)]] * 5 + [[]] * 5 + [[_box(100, 100)]] * 5
    _run(c, frames)
    assert c.total == 1


def test_long_gap_counts_again():
    c = PersonCounter(min_hits=2, max_age_sec=0.5)
    frames = [[_box(100, 100)]] * 5 + [[]] * 10 + [[_box(100, 100)]] * 5
    _run(c, frames)
    assert c.total == 2


def test_slow_inference_does_not_recount():
    # Inferencia lenta (CPU / Jetson caliente): 3 s entre frames, más que
    # max_age_sec. La misma persona, vista en todos los frames, cuenta UNA vez.
    c = PersonCounter(min_hits=3, max_age_sec=1.5)
    _run(c, [[_box(100, 200), _box(500, 200)] for _ in range(20)], dt=3.0)
    assert c.total == 2


def test_slow_inference_person_leaves_is_forgotten():
    c = PersonCounter(min_hits=2, max_age_sec=1.5, max_missed=3)
    frames = [[_box(100, 100)]] * 3 + [[]] * 3 + [[_box(100, 100)]] * 3
    _run(c, frames, dt=3.0)
    assert c.total == 2


def test_far_person_only_in_tiled_frames_counted_once():
    # Persona lejana: solo la detectan los tiles, 1 de cada 4 frames
    # (tiling.every_n_frames: 4), con inferencia lenta. person_detector pasa
    # max_missed * every_n_frames para que no se la olvide entre tiles.
    c = PersonCounter(min_hits=3, max_age_sec=1.5, max_missed=3 * 4)
    frames = [[_box(800, 300, w=12, h=24)] if i % 4 == 0 else [] for i in range(40)]
    _run(c, frames, dt=1.0)
    assert c.total == 1


def test_fast_motion_tracked_with_velocity():
    # Caja chica (persona lejana) que se corre 20 px por frame: el IoU entre
    # frames es 0, la predicción por velocidad la sigue igual.
    c = PersonCounter(min_hits=3)
    _run(c, [[_box(100 + 20 * i, 300, w=12, h=24)] for i in range(20)])
    assert c.total == 1


def test_new_ids_only_once_per_person():
    c = PersonCounter(min_hits=3)
    new = []
    for i in range(10):
        boxes = [_box(100, 200)] + ([_box(500, 200)] if i >= 4 else [])
        c.update(np.array(boxes), i * 0.1)
        new.append(list(c.new_ids))
    assert new[2] == [1]                 # 1ra persona: confirmada en el 3er frame
    assert new[6] == [2]                 # 2da: aparece en el frame 4, confirma en el 6
    assert sum(len(n) for n in new) == 2


def test_reset():
    c = PersonCounter(min_hits=1)
    _run(c, [[_box(100, 100)]])
    assert c.total == 1
    c.reset()
    assert c.total == 0 and not c.tracks
    ids = c.update(np.array([_box(100, 100)]), 10.0)
    assert ids[0] == 2                   # el id 1 ya está en el mapa: no se reutiliza
    assert c.total == 1


def test_noisy_velocity_after_gap_keeps_id():
    # Un salto de la caja deja una velocidad grande; tras un hueco la predicción
    # queda lejos, pero la persona sigue donde estaba: no es alguien nuevo.
    c = PersonCounter(min_hits=2, max_age_sec=1.5)
    frames = ([[_box(100, 200)]] * 3 + [[_box(140, 200)]] + [[]] * 10
              + [[_box(140, 200)]] * 3)
    _run(c, frames)
    assert c.total == 1


def test_empty_input():
    c = PersonCounter()
    assert len(c.update(np.zeros((0, 6)), 0.0)) == 0
    assert len(c.update([], 0.1)) == 0
