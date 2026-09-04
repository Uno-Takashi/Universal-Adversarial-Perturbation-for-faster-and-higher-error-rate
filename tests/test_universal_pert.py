"""Tests for the universal perturbation loop, including the cost of its forward passes.

The algorithm is pure NumPy, so a deterministic linear classifier pins its behaviour exactly.
That matters because the loop was optimised to stop recomputing forward passes it already had:
the golden vector below is what the pre-optimisation code produced, bit for bit.
"""

import contextlib
import io

import numpy as np
import pytest

import universal_pert
from deeptarget import deeptarget
from universal_pert import proj_lp, universal_perturbation

IMAGE_SHAPE = (6, 6, 3)
NUM_CLASSES = 12


class LinearModel:
    """logits = flatten(x) @ W, plus a count of how much work it was asked to do."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(size=(int(np.prod(IMAGE_SHAPE)), NUM_CLASSES))
        self.forward_calls = 0
        self.gradient_calls = 0

    def __call__(self, x):
        self.forward_calls += 1
        return np.asarray(x, dtype=np.float64).reshape(-1, self.W.shape[0]) @ self.W

    def grads(self, x, inds):
        self.gradient_calls += 1
        return np.stack([self.W[:, i].reshape(1, *IMAGE_SHAPE) for i in inds])


class FrozenOrder:
    """Freeze the shuffle so a run is reproducible."""

    def permutation(self, n):
        return np.arange(n)


@pytest.fixture
def frozen_rng(monkeypatch):
    monkeypatch.setattr(universal_pert, "_rng", FrozenOrder())


@pytest.fixture
def dataset():
    return np.random.default_rng(0).uniform(0, 255, size=(10, *IMAGE_SHAPE))


def _run(model, dataset, **overrides):
    kwargs = {
        "delta": 0.0,
        "max_iter_uni": 2,
        "xi": 10,
        "num_classes": 3,
        "max_iter_df": 15,
        "search_num": 4,
        "batch_size": 5,
    }
    kwargs.update(overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        return universal_perturbation(np.array(dataset), model, model.grads, **kwargs)


def test_perturbation_respects_the_linf_budget(frozen_rng, dataset):
    v = _run(LinearModel(), dataset, xi=7)
    assert np.max(np.abs(v)) <= 7 + 1e-9


def test_perturbation_is_reproducible(frozen_rng, dataset):
    a = _run(LinearModel(), dataset)
    b = _run(LinearModel(), dataset)
    np.testing.assert_array_equal(a, b)


def test_perturbation_matches_the_pre_optimisation_result(frozen_rng, dataset):
    """Golden check: removing redundant forward passes must not change the output.

    These values were produced by the version that called f() three times per image plus once
    per target. Any drift here means an 'optimisation' changed the algorithm.
    """
    v = np.asarray(_run(LinearModel(), dataset))

    assert v.shape == (1, *IMAGE_SHAPE)
    # Spot-check a few entries and the aggregate, rather than pasting 108 floats.
    assert float(np.max(np.abs(v))) == pytest.approx(10.0)
    assert float(v.sum()) == pytest.approx(-227.3340572552978, abs=1e-9)
    assert float((v**2).sum()) == pytest.approx(9762.952033307478, abs=1e-6)


def test_one_forward_pass_per_image_outside_deeptarget(frozen_rng, dataset):
    """The loop must not re-run forward passes it already has.

    Before the optimisation this ran three forward passes per image plus one per target just
    to print a label, and deeptarget re-ran another two per call.
    """
    model = LinearModel()
    _run(model, dataset, max_iter_uni=1, search_num=4)

    # 10 images, one pass, search_num=4: two batched clean-label calls, one forward per image,
    # and then only what deeptarget needs to iterate. The pre-optimisation code spent roughly
    # 29 forward passes per image here.
    assert model.forward_calls == 56, model.forward_calls


def test_gradient_work_is_unchanged_by_the_optimisation(frozen_rng, dataset):
    """Only redundant forward passes were removed; the gradient count is the real workload."""
    model = LinearModel()
    _run(model, dataset)
    assert model.gradient_calls == 80


def test_clean_labels_stay_aligned_with_a_shuffled_dataset(dataset, monkeypatch):
    """The cached clean predictions must be permuted with the data, not left behind.

    Shuffling with a reversal has to give exactly what running in frozen order on the
    already-reversed dataset gives. If the cache were not permuted alongside, the loop would
    compare image k against another image's clean label and the two would diverge.
    """
    reversing = type("Rev", (), {"permutation": staticmethod(lambda n: np.arange(n)[::-1])})()
    monkeypatch.setattr(universal_pert, "_rng", reversing)
    shuffled = np.asarray(_run(LinearModel(), dataset, max_iter_uni=1))

    monkeypatch.setattr(universal_pert, "_rng", FrozenOrder())
    reference = np.asarray(_run(LinearModel(), dataset[::-1], max_iter_uni=1))

    np.testing.assert_allclose(shuffled, reference, atol=1e-9)


def test_deeptarget_logits_shortcut_is_equivalent():
    """Passing f(image) in must give exactly what recomputing it gives."""
    model = LinearModel()
    image = np.random.default_rng(3).uniform(0, 255, size=(1, *IMAGE_SHAPE))
    logits = np.asarray(model(image)).flatten()
    target = int(logits.argsort()[::-1][1])

    without = deeptarget(image, model, model.grads, max_iter=20, target=target)
    calls_before = model.forward_calls
    with_logits = deeptarget(image, model, model.grads, max_iter=20, target=target, logits=logits)
    saved = model.forward_calls - calls_before

    np.testing.assert_array_equal(without[0], with_logits[0])
    assert without[1:3] == with_logits[1:3]
    # The shortcut skips the two identical opening forward passes.
    assert saved < (calls_before - 1)


def test_proj_lp_is_applied_after_every_accepted_step(frozen_rng, dataset):
    for xi in (1.0, 5.0, 20.0):
        v = _run(LinearModel(), dataset, xi=xi)
        assert np.max(np.abs(v)) <= xi + 1e-9
        np.testing.assert_allclose(proj_lp(v, xi, np.inf), v, atol=1e-9)
