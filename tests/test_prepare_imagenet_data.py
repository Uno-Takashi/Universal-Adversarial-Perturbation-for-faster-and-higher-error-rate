import numpy as np
import pytest
from PIL import Image

from prepare_imagenet_data import (
    CHANNEL_MEANS,
    do_image_avg,
    do_image_list,
    preprocess_image_batch,
    undo_image_avg,
    undo_image_list,
)


@pytest.fixture
def png_path(tmp_path):
    rng = np.random.default_rng(3)
    array = rng.integers(0, 256, size=(300, 200, 3), dtype=np.uint8)
    path = tmp_path / "img.png"
    Image.fromarray(array).save(path)
    return path


def test_preprocess_image_batch_resizes_and_crops(png_path):
    batch = preprocess_image_batch(
        [png_path], img_size=(256, 256), crop_size=(224, 224), color_mode="rgb"
    )

    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32


def test_preprocess_image_batch_subtracts_the_channel_means(png_path):
    batch = preprocess_image_batch([png_path], img_size=(8, 8))
    raw = np.asarray(Image.open(png_path).convert("RGB").resize((8, 8), Image.BILINEAR))

    expected = raw.astype("float32") - np.array(CHANNEL_MEANS, dtype="float32")
    np.testing.assert_allclose(batch[0], expected, atol=1e-4)


def test_preprocess_image_batch_appends_to_out(png_path):
    out = []
    result = preprocess_image_batch([png_path], img_size=(8, 8), out=out)

    assert result is None
    assert len(out) == 1
    assert out[0].shape == (1, 8, 8, 3)


def test_preprocess_image_batch_rejects_mismatched_shapes(tmp_path):
    paths = []
    for i, size in enumerate([(10, 10), (20, 20)]):
        path = tmp_path / f"img{i}.png"
        Image.fromarray(np.zeros((*size, 3), dtype=np.uint8)).save(path)
        paths.append(path)

    with pytest.raises(ValueError, match="same shapes"):
        preprocess_image_batch(paths)


def test_do_and_undo_image_avg_round_trip():
    rng = np.random.default_rng(4)
    img = rng.uniform(0, 255, size=(5, 5, 3)).astype(np.float32)

    np.testing.assert_allclose(undo_image_avg(do_image_avg(img)), img, atol=1e-3)


def test_image_list_helpers_round_trip():
    rng = np.random.default_rng(5)
    imgs = rng.integers(0, 256, size=(3, 5, 5, 3)).astype(np.float32)

    np.testing.assert_allclose(undo_image_list(do_image_list(imgs)), imgs, atol=1.0)
