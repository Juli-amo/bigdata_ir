# tests/test_iter_images.py
from pathlib import Path

from cli import iter_images


def test_iter_images_filters_and_recurses(tmp_path: Path):
    (tmp_path / "a.JPG").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "note.txt").write_text("nope")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.jpeg").write_bytes(b"x")

    got = sorted(p.name for p in iter_images(tmp_path))
    assert got == ["a.JPG", "b.png", "c.jpeg"]


def test_iter_images_single_file(tmp_path: Path):
    img = tmp_path / "alone.jpg"
    img.write_bytes(b"x")
    got = list(iter_images(img))
    assert [p.name for p in got] == ["alone.jpg"]
