import numpy as np
import pytest

from prepare_imagenet_data import undo_image_avg, undo_image_list
from util_univ import (
    avg_add_clip_pert,
    cat2label_str,
    clip_perturbed,
    fooling_rate_calc,
    fooling_rate_calc_all,
    target_fooling_rate_calc,
)


def _constant_classifier(label, num_classes=4):
    """Return an ``f`` that predicts ``label`` for every image in a batch."""

    def f(batch):
        logits = np.zeros((np.shape(batch)[0], num_classes))
        logits[:, label] = 1.0
        return logits

    return f


def test_avg_add_clip_pert_keeps_images_in_uint8_range():
    # A mean-subtracted image at the extremes, plus a perturbation that would overflow.
    img = np.full((1, 4, 4, 3), 120.0)
    v = np.full((1, 4, 4, 3), 500.0)

    pert_img = avg_add_clip_pert(img, v)

    restored = undo_image_avg(pert_img[0])
    assert restored.max() <= 255.0 + 1e-6
    assert restored.min() >= 0.0 - 1e-6


def test_target_fooling_rate_is_one_when_every_image_lands_on_target():
    dataset = np.zeros((7, 4, 4, 3))
    f = _constant_classifier(label=2)

    rate = target_fooling_rate_calc(v=1.0, dataset=dataset, f=f, target=2, batch_size=3)

    assert rate == 1.0


def test_target_fooling_rate_is_zero_for_a_different_target():
    dataset = np.zeros((7, 4, 4, 3))
    f = _constant_classifier(label=2)

    rate = target_fooling_rate_calc(v=1.0, dataset=dataset, f=f, target=3, batch_size=3)

    assert rate == 0.0


def test_fooling_rate_calc_all_is_zero_for_a_constant_classifier():
    dataset = np.zeros((5, 4, 4, 3))
    f = _constant_classifier(label=1)

    fooling_rate, target_rate = fooling_rate_calc_all(
        v=1.0, dataset=dataset, f=f, target=1, batch_size=2
    )

    assert fooling_rate == 0.0
    assert target_rate == 1.0


def test_fooling_rate_calc_all_detects_a_perturbation_that_flips_every_label():
    dataset = np.zeros((6, 4, 4, 3))
    v = np.ones((1, 4, 4, 3))

    def f(batch):
        # Predict class 0 for all-zero images and class 1 otherwise.
        logits = np.zeros((np.shape(batch)[0], 2))
        for i, img in enumerate(batch):
            logits[i, 0 if np.all(img == 0) else 1] = 1.0
        return logits

    fooling_rate, target_rate = fooling_rate_calc_all(
        v=v, dataset=dataset, f=f, target=1, batch_size=4
    )

    assert fooling_rate == 1.0
    assert target_rate == 1.0


@pytest.mark.parametrize(
    "num_pert",
    [
        1,
        np.int64(1),
        np.array([1]),  # what demo_inception passes: np.argmax(...).flatten()
        np.array(1),
        np.array([1.0]),
    ],
    ids=["int", "np-scalar", "1d-array", "0d-array", "float-array"],
)
def test_cat2label_str_accepts_scalars_and_single_element_arrays(num_pert, monkeypatch, tmp_path):
    labels = tmp_path / "data"
    labels.mkdir()
    (labels / "labels.txt").write_text("kit fox, Vulpes macrotis\nEnglish setter\n")
    monkeypatch.chdir(tmp_path)

    assert cat2label_str(num_pert) == "kit fox"


def test_clip_perturbed_keeps_pixels_in_range():
    dataset = np.full((2, 4, 4, 3), 250.0)
    v = np.full((1, 4, 4, 3), 20.0)

    clipped = clip_perturbed(dataset, v)

    assert clipped.max() == 255.0
    assert clipped.min() >= 0.0


def test_clip_perturbed_leaves_in_range_values_alone():
    dataset = np.full((2, 4, 4, 3), 100.0)
    v = np.full((1, 4, 4, 3), 10.0)

    np.testing.assert_allclose(clip_perturbed(dataset, v), 110.0)


def test_fooling_rate_calc_uses_one_preprocessing_convention():
    """Both halves of the comparison must see the same input convention.

    The pre-fix version fed f() mean-restored uint8 images while the perturbation had been
    built in mean-subtracted space, so a model that only ever sees consistent input would
    disagree with itself.
    """
    seen = []

    def f(batch):
        seen.append(np.copy(batch))
        logits = np.zeros((len(batch), 2))
        logits[:, 0] = 1.0
        return logits

    dataset = np.full((4, 4, 4, 3), 100.0)
    rate = fooling_rate_calc(v=0.0, dataset=dataset, f=f, batch_size=2)

    assert rate == 0.0
    # Every batch handed to f is the raw dataset, untouched.
    for batch in seen:
        assert batch.dtype == dataset.dtype
        np.testing.assert_allclose(batch, 100.0)


def test_fooling_rate_calc_clip_flag_changes_the_criterion():
    def f(batch):
        logits = np.zeros((len(batch), 2))
        for i, image in enumerate(batch):
            logits[i, 1 if image.mean() > 260 else 0] = 1.0
        return logits

    dataset = np.full((4, 4, 4, 3), 250.0)
    v = np.full((1, 4, 4, 3), 30.0)  # 280 unclipped, 255 clipped

    assert fooling_rate_calc(v, dataset, f, batch_size=2, clip=False) == 1.0
    assert fooling_rate_calc(v, dataset, f, batch_size=2, clip=True) == 0.0


def test_undo_image_list_clips_instead_of_wrapping():
    """An overshoot must saturate at 255, not wrap round to 44."""
    from prepare_imagenet_data import do_image_avg

    raw = np.array([[[300.0, 10.0, -50.0]]], dtype=np.float32)
    mean_subtracted = do_image_avg(raw)[None]

    restored = undo_image_list(mean_subtracted)

    np.testing.assert_array_equal(restored.flatten(), [255, 10, 0])
