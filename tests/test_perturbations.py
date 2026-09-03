"""Tests for the perturbation algorithms, driven by a synthetic linear classifier.

The classifier is ``logits(x) = W @ x.flatten()``, so its Jacobian is constant and can be
returned exactly. That makes deepfool/deeptarget exercisable without TensorFlow.
"""

import numpy as np
import pytest

from deepfool import deepfool
from deeptarget import deeptarget
from universal_pert import proj_lp

IMG_SHAPE = (1, 4, 4, 3)
NUM_CLASSES = 5


@pytest.fixture
def linear_model():
    """Return ``(f, grads, W)`` for a fixed random linear classifier."""
    rng = np.random.default_rng(0)
    n_features = int(np.prod(IMG_SHAPE))
    weights = rng.normal(size=(NUM_CLASSES, n_features))

    def f(image):
        return weights @ np.asarray(image).flatten()

    def grads(image, inds):
        # One gradient per requested class, each shaped like the image.
        return np.stack([weights[i].reshape(IMG_SHAPE) for i in inds])

    return f, grads, weights


def test_deepfool_changes_the_predicted_label(linear_model):
    f, grads, _ = linear_model
    image = np.zeros(IMG_SHAPE)
    image.flat[0] = 1.0

    original = int(np.argmax(np.array(f(image)).flatten()))
    r_tot, loop_i, k_i, pert_image = deepfool(image, f, grads, num_classes=NUM_CLASSES, max_iter=50)

    assert loop_i < 50, "deepfool should converge on a linear model"
    assert k_i != original
    assert r_tot.shape == IMG_SHAPE
    assert pert_image.shape == IMG_SHAPE
    assert int(np.argmax(np.array(f(pert_image)).flatten())) == k_i


def test_deeptarget_moves_towards_the_requested_class(linear_model):
    f, grads, _ = linear_model
    image = np.zeros(IMG_SHAPE)
    image.flat[0] = 1.0

    logits = np.array(f(image)).flatten()
    order = logits.argsort()[::-1]
    original, target = int(order[0]), int(order[1])

    r_tot, loop_i, k_i, _ = deeptarget(image, f, grads, max_iter=50, target=target)

    assert loop_i < 50
    assert k_i != original
    assert r_tot.shape == IMG_SHAPE
    # The runner-up class is the one the decision boundary is crossed into.
    assert k_i == target


def test_deeptarget_requires_a_target(linear_model):
    f, grads, _ = linear_model
    with pytest.raises(ValueError, match="Target"):
        deeptarget(np.zeros(IMG_SHAPE), f, grads)


@pytest.mark.parametrize("xi", [0.5, 1.0, 10.0])
def test_proj_lp_inf_respects_the_radius(xi):
    rng = np.random.default_rng(1)
    v = rng.normal(scale=5.0, size=IMG_SHAPE)

    projected = proj_lp(v, xi, np.inf)

    assert np.max(np.abs(projected)) <= xi + 1e-9
    assert np.all(np.sign(projected) == np.sign(v))


def test_proj_lp_l2_respects_the_radius():
    rng = np.random.default_rng(2)
    v = rng.normal(scale=5.0, size=IMG_SHAPE)

    projected = proj_lp(v, 1.0, 2)

    assert np.linalg.norm(projected.flatten()) <= 1.0 + 1e-9


def test_proj_lp_l2_leaves_small_vectors_untouched():
    v = np.full(IMG_SHAPE, 1e-4)

    projected = proj_lp(v, 100.0, 2)

    np.testing.assert_allclose(projected, v)


def test_proj_lp_rejects_other_norms():
    with pytest.raises(ValueError, match="not supported"):
        proj_lp(np.zeros(IMG_SHAPE), 1.0, 1)
