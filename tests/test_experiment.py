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


def test_print_table_renders_every_row(capsys):
    results = [
        {
            "model": "inception5h",
            "search_num": 5,
            "num_images": 20,
            "clean_top1": None,
            "fooling_rate": 0.9,
            "fooling_rate_clipped": 0.85,
            "seconds": 12.0,
        },
        {
            "model": "resnet50",
            "search_num": 5,
            "num_images": 20,
            "clean_top1": 0.75,
            "fooling_rate": 1.0,
            "fooling_rate_clipped": 1.0,
            "seconds": 30.0,
        },
    ]

    print_table(results)
    out = capsys.readouterr().out

    assert "inception5h" in out and "resnet50" in out
    assert "-" in out  # missing clean accuracy renders as a dash
    assert "75.0%" in out
    assert "90.0%" in out


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["experiment.py"])
    args = parse_args()

    assert args.models == ["inception5h"]
    assert args.search_num == [5]
    assert args.num_images == 20
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
