import numpy as np

from prepare_imagenet_data import undo_image_avg
from util_univ import (
    avg_add_clip_pert,
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
