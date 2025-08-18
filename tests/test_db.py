# Checks that cli.iter_images yields only supported image files from file or folder.

from pathlib import Path

from cli import iter_images


def test_iter_images_dir_and_file(tmp_path):
    img_dir = tmp_path / "d"
    img_dir.mkdir()
    # make 2 images and 1 non-image
    (img_dir / "a.jpg").write_bytes(b"fakejpeg")
    (img_dir / "b.png").write_bytes(b"fakepng")
    (img_dir / "note.txt").write_text("nope")

    # directory
    files = list(iter_images(img_dir))
    names = {p.name for p in files}
    assert "a.jpg" in names
    assert "b.png" in names
    assert "note.txt" not in names

    # single file
    single = list(iter_images(img_dir / "a.jpg"))
    assert len(single) == 1 and single[0].name == "a.jpg"