"""Tests for the experiment driver, using stub classifiers rather than real models."""

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import experiment
from experiment import evaluate, parse_args, print_table, run_transfer, top1_accuracy


class StubModel:
    """A classifier-shaped stub whose prediction depends only on whether v was added."""

    def __init__(self, num_classes=1000, image_size=(224, 224), offset=0, flip=True):
        self.num_classes = num_classes
        self.image_size = image_size
        self._offset = offset
        self._flip = flip

    def __call__(self, images):
        logits = np.zeros((len(images), self.num_classes))
        for i, image in enumerate(images):
            perturbed = self._flip and not np.allclose(image, 0.0)
            logits[i, 1 if perturbed else 0] = 1.0
        return logits

    def predict(self, images):
        return np.argmax(self(images), axis=1).flatten()

    def gradients(self, images, indices):
        return np.zeros((len(indices), *np.shape(images)))

    def to_ilsvrc(self, indices):
        return np.asarray(indices) - self._offset


def test_top1_accuracy_returns_none_without_labels():
    dataset = np.zeros((4, 4, 4, 3))
    assert top1_accuracy(StubModel(), dataset, np.asarray([], dtype=np.int64), 2) is None


def test_top1_accuracy_scores_against_ilsvrc_labels():
    dataset = np.zeros((4, 4, 4, 3))
    # The stub predicts class 0 for all-zero images.
    assert top1_accuracy(StubModel(), dataset, np.zeros(4, dtype=np.int64), 2) == 1.0
    assert top1_accuracy(StubModel(), dataset, np.ones(4, dtype=np.int64), 2) == 0.0


def test_top1_accuracy_applies_the_model_index_offset():
    """A model with its own class ordering is translated before being scored."""
    dataset = np.zeros((4, 4, 4, 3))
    shifted = StubModel(num_classes=1008, offset=-1)  # predicts 0, maps to ILSVRC 1
    assert top1_accuracy(shifted, dataset, np.ones(4, dtype=np.int64), 2) == 1.0


def test_top1_accuracy_batches_without_changing_the_answer():
    dataset = np.zeros((7, 4, 4, 3))
    labels = np.zeros(7, dtype=np.int64)
    assert top1_accuracy(StubModel(), dataset, labels, 1) == 1.0
    assert top1_accuracy(StubModel(), dataset, labels, 100) == 1.0


def test_evaluate_reports_both_clipped_and_unclipped_rates():
    dataset = np.zeros((5, 4, 4, 3))
    v = np.ones((1, 4, 4, 3)) * 5.0

    rates = evaluate(StubModel(), dataset, v, batch_size=2)

    assert set(rates) == {"fooling_rate", "fooling_rate_clipped"}
    assert rates["fooling_rate"] == 1.0
    assert rates["fooling_rate_clipped"] == 1.0


def test_evaluate_reports_zero_for_a_model_that_never_changes_its_mind():
    dataset = np.zeros((5, 4, 4, 3))
    v = np.ones((1, 4, 4, 3))

    rates = evaluate(StubModel(flip=False), dataset, v, batch_size=2)

    assert rates["fooling_rate"] == 0.0
    assert rates["fooling_rate_clipped"] == 0.0


def test_evaluate_clipping_can_neutralise_a_perturbation():
    """A perturbation that clipping removes entirely must show a lower clipped rate."""
    dataset = np.zeros((4, 4, 4, 3))
    v = np.full((1, 4, 4, 3), -50.0)  # clip(0 - 50, 0, 255) == 0, i.e. unchanged

    rates = evaluate(StubModel(), dataset, v, batch_size=2)

    assert rates["fooling_rate"] == 1.0
    assert rates["fooling_rate_clipped"] == 0.0


def test_run_transfer_skips_self_and_mismatched_input_sizes(capsys):
    dataset = np.zeros((4, 224, 224, 3))
    v224 = np.ones((1, 224, 224, 3))
    entries = {
        "a": ({}, StubModel(image_size=(224, 224)), dataset, v224),
        "b": ({}, StubModel(image_size=(224, 224)), dataset, v224),
        "c": ({}, StubModel(image_size=(299, 299)), np.zeros((4, 299, 299, 3)), v224),
    }

    rows = run_transfer(entries, batch_size=2)
    pairs = {(r["source"], r["target"]) for r in rows}

    assert ("a", "a") not in pairs
    assert ("a", "b") in pairs and ("b", "a") in pairs
    # c expects 299x299 input, so a 224x224 perturbation is not applied to it.
    assert not any(target == "c" for _, target in pairs)


def _result(model, **overrides):
    base = {
        "model": model,
        "search_num": 5,
        "num_images": 20,
        "num_val_images": 200,
        "clean_top1_val": 0.75,
        "train_fooling_rate": 1.0,
        "train_fooling_rate_clipped": 1.0,
        "val_fooling_rate": 0.4,
        "val_fooling_rate_clipped": 0.4,
        "val_random_baseline_clipped": 0.1,
        "seconds": 30.0,
    }
    base.update(overrides)
    return base


def test_print_table_renders_every_row(capsys):
    print_table([_result("inception5h", clean_top1_val=None), _result("resnet50")])
    out = capsys.readouterr().out

    assert "inception5h" in out and "resnet50" in out
    assert "-" in out  # missing clean accuracy renders as a dash
    assert "75.0%" in out
    assert "40.0%" in out  # the validation rate


def test_print_table_warns_that_the_generation_rate_is_circular(capsys):
    """The table must not let a reader mistake the fitted rate for evidence of universality."""
    print_table([_result("resnet50")])
    out = capsys.readouterr().out

    assert "stopping criterion" in out
    assert "val fool" in out and "random" in out


def test_random_baseline_saturates_the_budget(monkeypatch):
    seen = []

    def spy(v, dataset, f, batch_size=100, clip=False):
        seen.append(np.copy(v))
        return 0.25

    monkeypatch.setattr(experiment, "fooling_rate_calc", spy)

    rate = experiment.random_baseline(
        StubModel(), np.zeros((4, 8, 8, 3)), (1, 8, 8, 3), xi=10.0, batch_size=2, repeats=3
    )

    assert rate == 0.25
    assert len(seen) == 3
    for v in seen:
        assert v.shape == (1, 8, 8, 3)
        assert set(np.unique(v)) <= {-10.0, 10.0}


def test_random_baseline_averages_repeats(monkeypatch):
    rates = iter([0.0, 0.5, 1.0])
    monkeypatch.setattr(experiment, "fooling_rate_calc", lambda *a, **k: next(rates))
    value = experiment.random_baseline(
        StubModel(), np.zeros((2, 4, 4, 3)), (1, 4, 4, 3), xi=1.0, batch_size=2, repeats=3
    )
    assert value == pytest.approx(0.5)


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["experiment.py"])
    args = parse_args()

    assert args.models == ["inception5h"]
    assert args.search_num == [5]
    assert args.num_images == 100
    assert args.xi == 10.0
    assert args.transfer is False


def test_parse_args_accepts_multiple_models_and_a_sweep(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiment.py",
            "--models",
            "inception5h",
            "resnet50",
            "--search-num",
            "1",
            "3",
            "5",
            "--num-images",
            "8",
            "--transfer",
            "--out",
            "results",
        ],
    )
    args = parse_args()

    assert args.models == ["inception5h", "resnet50"]
    assert args.search_num == [1, 3, 5]
    assert args.num_images == 8
    assert args.transfer is True
    assert args.out == "results"


def test_module_exposes_a_main():
    assert callable(experiment.main)


class _SplitSpy:
    """Records exactly which images were fitted and which were scored."""

    def __init__(self):
        self.fitted = None
        self.scored = []

    def universal_perturbation(self, dataset, f, grads, **kwargs):
        # universal_perturbation shuffles in place, so snapshot before it can.
        self.fitted = np.copy(dataset)
        return np.zeros((1, *np.shape(dataset)[1:]), dtype=np.float32)

    def evaluate(self, model, dataset, v, batch_size):
        self.scored.append(np.copy(dataset))
        return {"fooling_rate": 0.0, "fooling_rate_clipped": 0.0}


@pytest.fixture
def split_args():
    class Args:
        dataset = "fake"
        num_images = 4
        num_val_images = 6
        seed = 0
        batch_size = 2
        delta = 0.2
        xi = 10.0
        num_classes = 2
        max_iter_uni = 1
        max_iter_df = 5
        out = None

    return Args()


def _fake_images(n, size=8):
    """Each image is a constant plane carrying its own index, so sets are identifiable."""
    images = np.stack([np.full((size, size, 3), float(i)) for i in range(n)])
    return images, np.arange(n) % 1000


def test_run_one_fits_and_scores_on_disjoint_images(monkeypatch, split_args):
    """The whole point: the reported rate must not come from the fitted images."""
    spy = _SplitSpy()
    monkeypatch.setattr(experiment, "build_classifier", lambda name: StubModel(image_size=(8, 8)))
    monkeypatch.setattr(experiment, "load_images", lambda *a, **k: _fake_images(k["num_images"]))
    monkeypatch.setattr(experiment, "universal_perturbation", spy.universal_perturbation)
    monkeypatch.setattr(experiment, "evaluate", spy.evaluate)
    monkeypatch.setattr(experiment, "random_baseline", lambda *a, **k: 0.05)

    result, _, val, _ = experiment.run_one("stub", split_args, search_num=3)

    fitted_ids = {int(img[0, 0, 0]) for img in spy.fitted}
    train_scored_ids = {int(img[0, 0, 0]) for img in spy.scored[0]}
    val_scored_ids = {int(img[0, 0, 0]) for img in spy.scored[1]}

    assert fitted_ids == {0, 1, 2, 3}
    assert train_scored_ids == fitted_ids, "the first evaluate call reports the fitted set"
    assert val_scored_ids == {4, 5, 6, 7, 8, 9}
    assert not (fitted_ids & val_scored_ids), "generation and validation must be disjoint"
    assert result["num_images"] == 4
    assert result["num_val_images"] == 6
    # Transfer must reuse the held-out images, not the fitted ones.
    assert {int(img[0, 0, 0]) for img in val} == val_scored_ids


def test_run_one_reports_train_and_val_separately(monkeypatch, split_args):
    monkeypatch.setattr(experiment, "build_classifier", lambda name: StubModel(image_size=(8, 8)))
    monkeypatch.setattr(experiment, "load_images", lambda *a, **k: _fake_images(k["num_images"]))
    monkeypatch.setattr(
        experiment,
        "universal_perturbation",
        lambda dataset, *a, **k: np.zeros((1, *np.shape(dataset)[1:]), dtype=np.float32),
    )
    rates = iter(
        [
            {"fooling_rate": 1.0, "fooling_rate_clipped": 1.0},
            {"fooling_rate": 0.3, "fooling_rate_clipped": 0.25},
        ]
    )
    monkeypatch.setattr(experiment, "evaluate", lambda *a, **k: next(rates))
    monkeypatch.setattr(experiment, "random_baseline", lambda *a, **k: 0.05)

    result, *_ = experiment.run_one("stub", split_args, search_num=3)

    assert result["train_fooling_rate"] == 1.0
    assert result["val_fooling_rate"] == 0.3
    assert result["val_fooling_rate_clipped"] == 0.25
    assert result["val_random_baseline_clipped"] == 0.05


def test_run_one_refuses_a_short_draw(monkeypatch, split_args):
    monkeypatch.setattr(experiment, "build_classifier", lambda name: StubModel(image_size=(8, 8)))
    monkeypatch.setattr(experiment, "load_images", lambda *a, **k: _fake_images(3))

    with pytest.raises(ValueError, match="got 3"):
        experiment.run_one("stub", split_args, search_num=3)


def test_clean_accuracy_is_measured_on_validation(monkeypatch, split_args):
    seen = {}
    monkeypatch.setattr(experiment, "build_classifier", lambda name: StubModel(image_size=(8, 8)))
    monkeypatch.setattr(experiment, "load_images", lambda *a, **k: _fake_images(k["num_images"]))
    monkeypatch.setattr(
        experiment,
        "universal_perturbation",
        lambda dataset, *a, **k: np.zeros((1, *np.shape(dataset)[1:]), dtype=np.float32),
    )
    monkeypatch.setattr(
        experiment, "evaluate", lambda *a, **k: {"fooling_rate": 0.0, "fooling_rate_clipped": 0.0}
    )
    monkeypatch.setattr(experiment, "random_baseline", lambda *a, **k: 0.0)

    def spy_accuracy(model, dataset, labels, batch_size):
        seen["ids"] = {int(img[0, 0, 0]) for img in dataset}
        return 0.5

    monkeypatch.setattr(experiment, "top1_accuracy", spy_accuracy)
    experiment.run_one("stub", split_args, search_num=3)

    assert seen["ids"] == {4, 5, 6, 7, 8, 9}


def test_parse_args_exposes_the_validation_size(monkeypatch):
    monkeypatch.setattr("sys.argv", ["experiment.py", "--num-val-images", "500"])
    assert parse_args().num_val_images == 500


def test_parse_args_validation_size_defaults_larger_than_generation(monkeypatch):
    monkeypatch.setattr("sys.argv", ["experiment.py"])
    args = parse_args()
    assert args.num_val_images > args.num_images


def test_default_generation_set_is_large_enough_to_generalise(monkeypatch):
    """Below roughly 100 images the perturbation overfits, so the default must not be lower."""
    monkeypatch.setattr("sys.argv", ["experiment.py"])
    assert parse_args().num_images >= 100
