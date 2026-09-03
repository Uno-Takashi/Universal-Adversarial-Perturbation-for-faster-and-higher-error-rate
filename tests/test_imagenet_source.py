"""Tests for the ImageNet loaders. The Hugging Face path is exercised against a fake HTTP layer."""

import io
import json

import numpy as np
import pytest
from PIL import Image

import imagenet_source
from imagenet_source import (
    _decode_image,
    _pick_column,
    load_hf_images,
    load_images,
    load_local_images,
    resize_and_crop,
)


def _png_bytes(size=(64, 48), colour=(10, 20, 30)):
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("image_size", "expected"),
    [((224, 224), (224, 224)), ((299, 299), (299, 299)), ((128, 96), (96, 128))],
    ids=["224", "299", "non-square"],
)
def test_resize_and_crop_produces_the_requested_size(image_size, expected):
    source = Image.new("RGB", (500, 375), (1, 2, 3))
    cropped = resize_and_crop(source, image_size)
    assert cropped.size == expected  # PIL reports (width, height)


def test_resize_and_crop_centres_the_crop():
    # A 4-pixel-wide white stripe down the middle stays in the middle after cropping.
    source = Image.new("RGB", (256, 256), (0, 0, 0))
    for x in range(126, 130):
        for y in range(256):
            source.putpixel((x, y), (255, 255, 255))

    cropped = np.asarray(resize_and_crop(source, (224, 224)))
    column_brightness = cropped.mean(axis=(0, 2))
    assert 100 < int(np.argmax(column_brightness)) < 124


def test_resize_and_crop_converts_greyscale_to_rgb():
    cropped = resize_and_crop(Image.new("L", (300, 300), 128), (32, 32))
    assert np.asarray(cropped).shape == (32, 32, 3)


def test_pick_column_prefers_known_names():
    assert _pick_column({"image": 1, "label": 2}, ("image", "img"), "image") == "image"
    assert _pick_column({"crop224": 1}, ("image", "crop224"), "image") == "crop224"


def test_pick_column_falls_back_to_a_src_dict():
    assert _pick_column({"weird": {"src": "http://x"}}, ("image",), "image") == "weird"


def test_pick_column_returns_none_when_absent():
    assert _pick_column({"a": 1}, ("label",), "label") is None


def test_decode_image_accepts_raw_bytes_and_byte_dicts():
    assert _decode_image(_png_bytes()).size == (64, 48)
    assert _decode_image({"bytes": _png_bytes()}).size == (64, 48)


def test_decode_image_rejects_other_types():
    with pytest.raises(TypeError):
        _decode_image(42)


class _FakeHttp:
    """Serve the dataset-viewer /rows contract from memory."""

    def __init__(self, num_rows=250, fail_times=0):
        self.num_rows = num_rows
        self.fail_times = fail_times
        self.calls = 0

    def __call__(self, url, attempts=5, backoff=2.0):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient")
        if url.startswith("image://"):
            return _png_bytes()
        offset = int(url.split("offset=")[1].split("&")[0])
        length = int(url.split("length=")[1].split("&")[0])
        rows = [
            {"row": {"image": {"src": f"image://{i}"}, "label": i % 1000}}
            for i in range(offset, min(offset + length, self.num_rows))
        ]
        return json.dumps({"num_rows_total": self.num_rows, "rows": rows}).encode()


def test_load_hf_images_returns_raw_pixels(monkeypatch):
    monkeypatch.setattr(imagenet_source, "_fetch", _FakeHttp())

    images, labels = load_hf_images("some/dataset", num_images=5, image_size=(32, 32), seed=None)

    assert images.shape == (5, 32, 32, 3)
    assert images.dtype == np.float32
    assert 0.0 <= images.min() and images.max() <= 255.0
    assert labels.tolist() == [0, 1, 2, 3, 4]


def test_load_hf_images_pages_past_the_request_cap(monkeypatch):
    fake = _FakeHttp(num_rows=250)
    monkeypatch.setattr(imagenet_source, "_fetch", fake)
    monkeypatch.setattr(imagenet_source, "MAX_ROWS_PER_REQUEST", 4)

    images, labels = load_hf_images("some/dataset", num_images=10, image_size=(16, 16), seed=None)

    assert images.shape == (10, 16, 16, 3)
    assert labels.tolist() == list(range(10))


def test_load_hf_images_seed_picks_a_random_window(monkeypatch):
    monkeypatch.setattr(imagenet_source, "_fetch", _FakeHttp(num_rows=1000))

    _, labels_a = load_hf_images("d", num_images=4, image_size=(16, 16), seed=1)
    _, labels_b = load_hf_images("d", num_images=4, image_size=(16, 16), seed=2)
    _, labels_a_again = load_hf_images("d", num_images=4, image_size=(16, 16), seed=1)

    assert labels_a.tolist() == labels_a_again.tolist(), "seeding must be reproducible"
    assert labels_a.tolist() != labels_b.tolist(), "different seeds must differ"


def test_load_hf_images_resolves_known_shorthands(monkeypatch):
    seen = []
    backing = _FakeHttp()

    def spy(url, **kwargs):
        seen.append(url)
        return backing(url, **kwargs)

    monkeypatch.setattr(imagenet_source, "_fetch", spy)
    load_hf_images("imagenet-val", num_images=1, image_size=(16, 16), seed=None)

    rows_urls = [u for u in seen if u.startswith(imagenet_source.ROWS_ENDPOINT)]
    assert rows_urls, seen
    assert all("ImageNet1K-val-indexed" in u for u in rows_urls)
    assert "split=train" in rows_urls[0]


def test_load_hf_images_rejects_a_request_larger_than_the_dataset(monkeypatch):
    monkeypatch.setattr(imagenet_source, "_fetch", _FakeHttp(num_rows=3))
    with pytest.raises(ValueError, match="only 3 rows"):
        load_hf_images("d", num_images=10, image_size=(16, 16))


def test_fetch_retries_transient_failures(monkeypatch):
    calls = {"n": 0}

    def flaky(url, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("slow")
        return io.BytesIO(b"payload")

    class _Response:
        def __init__(self, data):
            self._data = data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return self._data

    def urlopen(request, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("slow")
        return _Response(b"payload")

    monkeypatch.setattr(imagenet_source.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(imagenet_source.time, "sleep", lambda _: None)

    assert imagenet_source._fetch("http://x") == b"payload"
    assert calls["n"] == 3


def test_fetch_gives_up_and_reports(monkeypatch):
    def always_fail(request, timeout=None):
        raise TimeoutError("nope")

    monkeypatch.setattr(imagenet_source.urllib.request, "urlopen", always_fail)
    monkeypatch.setattr(imagenet_source.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="Gave up"):
        imagenet_source._fetch("http://x", attempts=2)


@pytest.fixture
def imagenet_tree(tmp_path):
    for class_index in range(4):
        class_dir = tmp_path / f"n{class_index:08d}"
        class_dir.mkdir()
        for image_index in range(3):
            Image.new("RGB", (300, 250), (class_index * 40, image_index * 40, 0)).save(
                class_dir / f"{image_index}.JPEG"
            )
    return tmp_path


def test_load_local_images_spreads_across_classes(imagenet_tree):
    images, labels = load_local_images(str(imagenet_tree), num_images=4, image_size=(32, 32))

    assert images.shape == (4, 32, 32, 3)
    assert labels.size == 0
    # One image per class directory, so every red channel value is distinct.
    assert len({round(float(img[..., 0].mean())) for img in images}) == 4


def test_load_local_images_honours_per_class(imagenet_tree):
    images, _ = load_local_images(
        str(imagenet_tree), num_images=12, per_class=2, image_size=(16, 16)
    )
    assert len(images) == 8


def test_load_local_images_skips_unreadable_files(imagenet_tree):
    (imagenet_tree / "n00000000" / "broken.JPEG").write_bytes(b"not an image")
    images, _ = load_local_images(str(imagenet_tree), num_images=4, image_size=(16, 16))
    assert len(images) == 4


def test_load_local_images_needs_class_directories(tmp_path):
    with pytest.raises(ValueError, match="No class subdirectories"):
        load_local_images(str(tmp_path))


def test_load_images_dispatches_on_the_source(imagenet_tree, monkeypatch):
    local, _ = load_images(str(imagenet_tree), num_images=2, image_size=(16, 16))
    assert local.shape == (2, 16, 16, 3)

    monkeypatch.setattr(imagenet_source, "_fetch", _FakeHttp())
    remote, _ = load_images("some/dataset", num_images=2, image_size=(16, 16), seed=None)
    assert remote.shape == (2, 16, 16, 3)


def test_cache_round_trip_avoids_a_second_fetch(monkeypatch, tmp_path):
    """A repeated request must be served from disk, not refetched."""
    monkeypatch.setenv("UAP_IMAGE_CACHE", str(tmp_path / "cache"))
    fake = _FakeHttp()
    monkeypatch.setattr(imagenet_source, "_fetch", fake)

    first_images, first_labels = load_hf_images("d", num_images=5, image_size=(16, 16), seed=None)
    calls_after_first = fake.calls

    second_images, second_labels = load_hf_images("d", num_images=5, image_size=(16, 16), seed=None)

    assert fake.calls == calls_after_first, "the second call must not hit the network"
    np.testing.assert_array_equal(first_images, second_images)
    np.testing.assert_array_equal(first_labels, second_labels)


def test_cache_writes_a_file_that_actually_lands(monkeypatch, tmp_path):
    """np.savez renames a *name* by appending .npz; the temp file must still be replaced."""
    cache = tmp_path / "cache"
    monkeypatch.setenv("UAP_IMAGE_CACHE", str(cache))
    monkeypatch.setattr(imagenet_source, "_fetch", _FakeHttp())

    load_hf_images("d", num_images=3, image_size=(16, 16), seed=None)

    files = sorted(p.name for p in cache.iterdir())
    assert len(files) == 1, files
    assert files[0].endswith(".npz")
    assert ".tmp" not in files[0], "the temporary file was never renamed"


def test_cache_keys_separate_distinct_requests(monkeypatch, tmp_path):
    monkeypatch.setenv("UAP_IMAGE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(imagenet_source, "_fetch", _FakeHttp())

    load_hf_images("d", num_images=3, image_size=(16, 16), seed=None)
    load_hf_images("d", num_images=4, image_size=(16, 16), seed=None)
    load_hf_images("d", num_images=3, image_size=(32, 32), seed=None)

    assert len(list((tmp_path / "cache").iterdir())) == 3


def test_use_cache_false_bypasses_the_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("UAP_IMAGE_CACHE", str(tmp_path / "cache"))
    fake = _FakeHttp()
    monkeypatch.setattr(imagenet_source, "_fetch", fake)

    load_hf_images("d", num_images=3, image_size=(16, 16), seed=None, use_cache=False)
    calls = fake.calls
    load_hf_images("d", num_images=3, image_size=(16, 16), seed=None, use_cache=False)

    assert fake.calls > calls
    assert not (tmp_path / "cache").exists()


def test_a_corrupt_cache_file_is_ignored(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    monkeypatch.setenv("UAP_IMAGE_CACHE", str(cache))
    monkeypatch.setattr(imagenet_source, "_fetch", _FakeHttp())

    load_hf_images("d", num_images=3, image_size=(16, 16), seed=None)
    cached_file = next(cache.iterdir())
    cached_file.write_bytes(b"not an npz")

    images, _ = load_hf_images("d", num_images=3, image_size=(16, 16), seed=None)
    assert images.shape == (3, 16, 16, 3)


def test_parallel_and_serial_fetching_agree(monkeypatch, tmp_path):
    monkeypatch.setenv("UAP_IMAGE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(imagenet_source, "_fetch", _FakeHttp())
    parallel, labels_p = load_hf_images(
        "d", num_images=8, image_size=(16, 16), seed=None, workers=8, use_cache=False
    )

    monkeypatch.setattr(imagenet_source, "_fetch", _FakeHttp())
    serial, labels_s = load_hf_images(
        "d", num_images=8, image_size=(16, 16), seed=None, workers=1, use_cache=False
    )

    np.testing.assert_array_equal(parallel, serial)
    np.testing.assert_array_equal(labels_p, labels_s)


def test_parallel_fetching_preserves_row_order(monkeypatch, tmp_path):
    """Threads must not reorder images relative to their labels."""
    monkeypatch.setenv("UAP_IMAGE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(imagenet_source, "_fetch", _FakeHttp())

    _, labels = load_hf_images(
        "d", num_images=20, image_size=(16, 16), seed=None, workers=8, use_cache=False
    )

    assert labels.tolist() == list(range(20))
