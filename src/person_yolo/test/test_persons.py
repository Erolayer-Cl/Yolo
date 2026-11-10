import math

from person_yolo.persons import Observation, PersonRegistry

LAT0, LON0 = -33.4670, -70.6570
M_PER_DEG = 111_195.0


def obs(tid, north_m=None, err=2.0):
    if north_m is None:
        return Observation(tid)
    return Observation(tid, LAT0 + north_m / M_PER_DEG, LON0, err)


def run(reg, frames, dt=0.1, t0=0.0):
    """frames: lista de listas de Observation. Devuelve todos los eventos."""
    events = []
    for i, f in enumerate(frames):
        t = t0 + i * dt
        events += reg.update(t, 1000.0 + t, f)
    return events


def kinds(events):
    return [e.kind for e in events]


def test_waypoint_only_after_min_seen():
    reg = PersonRegistry(min_seen_sec=2.0)
    ev = run(reg, [[obs(1, 10.0)]] * 15)                 # 1.4 s
    assert ev == [] and reg.total == 0 and reg.person_id(1) == 0
    ev = run(reg, [[obs(1, 10.0)]] * 10, t0=1.5)         # llega a 2.4 s
    assert kinds(ev) == ['alert'] and reg.total == 1 and reg.person_id(1) == 1


def test_short_false_positive_never_becomes_waypoint():
    reg = PersonRegistry(min_seen_sec=2.0, forget_sec=1.0)
    ev = run(reg, [[obs(1, 10.0)]] * 5 + [[]] * 30)
    assert ev == [] and reg.total == 0 and not reg.pending


def test_waits_for_a_position_but_not_forever():
    reg = PersonRegistry(min_seen_sec=2.0, max_wait_sec=4.0)
    ev = run(reg, [[obs(1)]] * 30)                       # 2.9 s sin posición: espera
    assert ev == []
    ev = run(reg, [[obs(1)]] * 15, t0=3.0)               # pasa los 4 s
    assert kinds(ev) == ['alert'] and not ev[0].obs.located


def test_position_found_while_waiting_is_used():
    reg = PersonRegistry(min_seen_sec=2.0)
    frames = [[obs(1)]] * 10 + [[obs(1, 10.0, err=6.0)]] * 5 + [[obs(1, 11.0, err=2.0)]] * 10
    ev = run(reg, frames)
    assert kinds(ev) == ['alert']
    # Fusión pesada por 1/error²: las de ±2 m mandan, y el error baja (con piso).
    north = (ev[0].obs.lat - LAT0) * M_PER_DEG
    assert 10.8 < north < 11.0
    assert 1.0 <= ev[0].obs.error_m < 2.0


def test_lost_after_min_seen_is_announced_anyway():
    reg = PersonRegistry(min_seen_sec=2.0, max_wait_sec=10.0, forget_sec=1.0)
    ev = run(reg, [[obs(1)]] * 25 + [[]] * 15)
    assert kinds(ev) == ['alert']


def test_same_person_back_in_frame_is_a_duplicate():
    reg = PersonRegistry(min_seen_sec=1.0, dedupe_radius_m=4.0)
    ev = run(reg, [[obs(1, 10.0)]] * 15 + [[]] * 20 + [[obs(2, 12.0)]] * 15)
    assert kinds(ev) == ['alert', 'duplicate']
    assert reg.total == 1 and reg.person_id(2) == 1


def test_duplicate_with_better_position_updates_it():
    reg = PersonRegistry(min_seen_sec=1.0)
    ev = run(reg, [[obs(1, 10.0, err=5.0)]] * 15 + [[]] * 20 + [[obs(2, 11.0, err=1.0)]] * 15)
    assert kinds(ev) == ['alert', 'duplicate', 'fix']
    assert ev[2].person_id == 1 and ev[2].t == ev[0].t


def test_two_people_seen_together_are_never_merged():
    reg = PersonRegistry(min_seen_sec=1.0, dedupe_radius_m=4.0)
    ev = run(reg, [[obs(1, 10.0), obs(2, 11.0)]] * 15)
    assert kinds(ev) == ['alert', 'alert'] and reg.total == 2


def test_far_away_person_is_new():
    reg = PersonRegistry(min_seen_sec=1.0, dedupe_radius_m=4.0)
    ev = run(reg, [[obs(1, 10.0)]] * 15 + [[]] * 20 + [[obs(2, 40.0)]] * 15)
    assert kinds(ev) == ['alert', 'alert'] and reg.person_id(2) == 2


def test_large_error_widens_duplicate_radius():
    reg = PersonRegistry(min_seen_sec=1.0, dedupe_radius_m=4.0)
    # 9 m de distancia, pero con errores de 4 m: 2 * hypot(4, 4) = 11.3 m
    ev = run(reg, [[obs(1, 10.0, err=4.0)]] * 15 + [[]] * 20 + [[obs(2, 19.0, err=4.0)]] * 15)
    assert kinds(ev)[:2] == ['alert', 'duplicate'] and reg.total == 1
    # Al absorberla, la posición queda en el medio: puede salir un fix.
    assert set(kinds(ev)[2:]) <= {'fix'}


def test_dedupe_can_be_disabled():
    reg = PersonRegistry(min_seen_sec=1.0, dedupe_radius_m=0.0)
    ev = run(reg, [[obs(1, 10.0)]] * 15 + [[]] * 20 + [[obs(2, 10.0)]] * 15)
    assert kinds(ev) == ['alert', 'alert']


def test_fix_only_on_clear_improvement():
    reg = PersonRegistry(min_seen_sec=1.0, min_ratio=0.8)
    frames = ([[obs(1, 10.0, err=5.0)]] * 15 + [[obs(1, 10.0, err=4.5)]] * 3
              + [[obs(1, 10.0, err=3.0)]] * 3 + [[obs(1)]] * 3)
    ev = run(reg, frames)
    assert kinds(ev) == ['alert', 'fix']               # un solo fix (como mucho 1 por s)
    assert ev[1].obs.error_m <= 0.8 * ev[0].obs.error_m
    assert ev[1].obs.error_m >= 0.5 * 3.0               # piso: mitad del mejor individual


def test_unlocated_person_gets_fix_later():
    reg = PersonRegistry(min_seen_sec=1.0, max_wait_sec=1.0)
    ev = run(reg, [[obs(1)]] * 15 + [[obs(1, 10.0)]] * 3)
    assert kinds(ev) == ['alert', 'fix']


def approx(tid, drone_north_m):
    """Persona sin ubicar, con el dron `drone_north_m` al norte de LAT0."""
    return Observation(tid, drone_lat=LAT0 + drone_north_m / M_PER_DEG, drone_lon=LON0)


def test_approx_with_drone_in_same_place_is_a_duplicate():
    # Dron quieto, persona lejana que parpadea: el tracker le da otro id cada vez.
    reg = PersonRegistry(min_seen_sec=1.0, max_wait_sec=1.0, approx_radius_m=10.0)
    ev = run(reg, [[approx(1, 0.0)]] * 15 + [[]] * 20 + [[approx(2, 1.0)]] * 15
             + [[]] * 20 + [[approx(3, 0.5)]] * 15)
    assert kinds(ev) == ['alert', 'duplicate', 'duplicate']
    assert reg.total == 1 and reg.person_id(3) == 1


def test_approx_seen_together_are_never_merged():
    reg = PersonRegistry(min_seen_sec=1.0, max_wait_sec=1.0, approx_radius_m=10.0)
    ev = run(reg, [[approx(1, 0.0), approx(2, 0.0), approx(3, 0.0)]] * 15
             + [[]] * 20 + [[approx(4, 0.0), approx(5, 0.0)]] * 15)
    # Tres a la vez = tres personas; las dos que vuelven son dos de esas tres.
    assert kinds(ev) == ['alert'] * 3 + ['duplicate'] * 2
    assert reg.total == 3 and reg.person_id(4) != reg.person_id(5)


def test_approx_with_drone_far_away_is_new():
    reg = PersonRegistry(min_seen_sec=1.0, max_wait_sec=1.0, approx_radius_m=10.0)
    ev = run(reg, [[approx(1, 0.0)]] * 15 + [[]] * 20 + [[approx(2, 30.0)]] * 15)
    assert kinds(ev) == ['alert', 'alert'] and reg.total == 2


def test_approx_dedupe_can_be_disabled_and_needs_gps():
    reg = PersonRegistry(min_seen_sec=1.0, max_wait_sec=1.0, approx_radius_m=0.0)
    ev = run(reg, [[approx(1, 0.0)]] * 15 + [[]] * 20 + [[approx(2, 0.0)]] * 15)
    assert kinds(ev) == ['alert', 'alert']
    reg = PersonRegistry(min_seen_sec=1.0, max_wait_sec=1.0, approx_radius_m=10.0)
    ev = run(reg, [[obs(1)]] * 15 + [[]] * 20 + [[obs(2)]] * 15)     # sin GPS del dron
    assert kinds(ev) == ['alert', 'alert']


def test_approx_never_merges_with_a_located_person():
    # Una ubicada tiene su propia posición, que puede estar lejos del dron.
    reg = PersonRegistry(min_seen_sec=1.0, max_wait_sec=1.0, approx_radius_m=10.0)
    located = Observation(1, LAT0, LON0, 2.0, drone_lat=LAT0, drone_lon=LON0)
    ev = run(reg, [[located]] * 15 + [[]] * 20 + [[approx(2, 0.0)]] * 15)
    assert kinds(ev) == ['alert', 'alert']


def located(tid, north_m, east_m=0.0, err=2.0):
    return Observation(tid, LAT0 + north_m / M_PER_DEG,
                       LON0 + east_m / (M_PER_DEG * math.cos(math.radians(LAT0))), err)


def north_of(o):
    return (o.lat - LAT0) * M_PER_DEG


def test_fusion_averages_noise_and_rejects_outliers():
    rnd = __import__('random').Random(1)
    frames = [[located(1, 10.0 + rnd.gauss(0, 2.0))] for _ in range(40)]
    frames[25] = [located(1, 60.0)]                     # un rayo rasante que salió mal
    reg = PersonRegistry(min_seen_sec=1.0)
    ev = run(reg, frames)
    last = [e for e in ev if e.kind in ('alert', 'fix')][-1]
    assert abs(north_of(last.obs) - 10.0) < 1.0         # una sola obs tiene ±2 m


def test_waits_for_good_position_up_to_max_wait():
    # Lejos (±15 m) y acercándose (±3 m): no se marca con ±15 si llega algo mejor.
    reg = PersonRegistry(min_seen_sec=1.0, max_wait_sec=5.0, max_error_m=8.0)
    ev = run(reg, [[located(1, 10.0, err=15.0)]] * 20 + [[located(1, 10.0, err=3.0)]] * 10)
    assert kinds(ev)[0] == 'alert' and kinds(ev).count('alert') == 1
    assert ev[0].obs.error_m < 8.0                     # no se marcó con los ±15 m
    reg = PersonRegistry(min_seen_sec=1.0, max_wait_sec=3.0, max_error_m=8.0)
    ev = run(reg, [[located(1, 10.0, err=15.0)]] * 40)    # nunca mejora: igual se marca
    assert kinds(ev) == ['alert'] and ev[0].obs.located


def test_two_marks_that_converge_are_merged_live():
    # La misma persona vista dos veces (tracks 1 y 2), mal ubicada la 1ra vez:
    # al mejorar, las dos quedan juntas y se fusionan en la más vieja.
    reg = PersonRegistry(min_seen_sec=1.0, dedupe_radius_m=4.0, fix_period_sec=0.0)
    ev = run(reg, [[located(1, 10.0, err=1.0)]] * 15 + [[]] * 25
             + [[located(2, 25.0, err=1.0)]] * 15 + [[located(2, 12.0, err=1.0)]] * 60)
    assert 'merge' in kinds(ev) and reg.total == 1
    m = next(e for e in ev if e.kind == 'merge')
    assert m.person_id == 1 and m.removed[0] == 2
    assert reg.person_id(2) == 1                       # el track sigue, con el id viejo


def test_people_seen_together_are_never_merged_later():
    reg = PersonRegistry(min_seen_sec=1.0, dedupe_radius_m=4.0, fix_period_sec=0.0)
    ev = run(reg, [[located(1, 10.0, err=1.0), located(2, 20.0, err=1.0)]] * 15
             + [[located(2, 11.0, err=1.0)]] * 60)       # la 2 "se acerca" a la 1
    assert 'merge' not in kinds(ev) and reg.total == 2


def test_reset_keeps_ids_unique():
    reg = PersonRegistry(min_seen_sec=1.0)
    run(reg, [[obs(1, 10.0)]] * 15)
    reg.reset()
    assert reg.total == 0
    ev = run(reg, [[obs(2, 10.0)]] * 15, t0=10.0)
    assert kinds(ev) == ['alert'] and ev[0].person_id == 2
    assert not math.isinf(ev[0].obs.error_m)
