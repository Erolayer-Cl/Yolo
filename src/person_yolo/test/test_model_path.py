from person_yolo.model_path import is_downloadable, resolve_model_path


def test_finds_in_models_subdir(tmp_path):
    (tmp_path / 'models').mkdir()
    (tmp_path / 'models' / 'yolo11n.pt').write_bytes(b'x')
    assert resolve_model_path('yolo11n.pt', [tmp_path]) == str(tmp_path / 'models' / 'yolo11n.pt')


def test_relative_path_with_folder_from_workspace_root(tmp_path):
    # Jetson: model: models/yolo11n.engine, lanzado desde otra carpeta.
    (tmp_path / 'models').mkdir()
    (tmp_path / 'models' / 'yolo11n.engine').write_bytes(b'x')
    elsewhere = tmp_path / 'otra'
    elsewhere.mkdir()
    got = resolve_model_path('models/yolo11n.engine', [elsewhere, tmp_path])
    assert got == str(tmp_path / 'models' / 'yolo11n.engine')


def test_first_root_wins(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    for r in (a, b):
        r.mkdir()
        (r / 'w.pt').write_bytes(b'x')
    assert resolve_model_path('w.pt', [a, b]) == str(a / 'w.pt')


def test_missing_returns_none(tmp_path):
    assert resolve_model_path('models/nada.engine', [tmp_path]) is None
    assert resolve_model_path(str(tmp_path / 'abs.pt'), [tmp_path]) is None


def test_is_downloadable():
    assert is_downloadable('yolo11n.pt')
    assert not is_downloadable('models/yolo11n.pt')
    assert not is_downloadable('yolo11n.engine')
