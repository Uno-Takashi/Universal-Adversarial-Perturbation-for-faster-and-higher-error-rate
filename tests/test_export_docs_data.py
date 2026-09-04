"""Tests for the exporter that feeds the documentation site."""

import json

import pytest

from export_docs_data import PARADIGMS, load_sweep


def _summary(tmp_path, sweep, model, num_images, **overrides):
    entry = {
        "model": model,
        "search_num": 5,
        "num_images": num_images,
        "num_val_images": 200,
        "xi": 10.0,
        "image_size": [224, 224],
        "clean_top1_val": 0.7,
        "seconds": 42.0,
        "perturbation_linf": 10.0,
        "train_fooling_rate": 0.9,
        "train_fooling_rate_clipped": 0.9,
        "val_fooling_rate": 0.4,
        "val_fooling_rate_clipped": 0.4,
        "val_random_baseline_clipped": 0.25,
    }
    entry.update(overrides)
    out = tmp_path / sweep / model / str(num_images)
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps({"args": {}, "results": [entry]}))
    return out


def test_load_sweep_flattens_nested_result_trees(tmp_path):
    _summary(tmp_path, "images", "resnet50", 128)
    _summary(tmp_path, "images", "resnet50", 256)
    _summary(tmp_path, "images", "mobilenet_v2", 128)

    runs = load_sweep(str(tmp_path / "images"), "images")

    assert len(runs) == 3
    assert {r["model"] for r in runs} == {"resnet50", "mobilenet_v2"}
    assert all(r["sweep"] == "images" for r in runs)


def test_margin_is_val_minus_baseline(tmp_path):
    _summary(
        tmp_path,
        "images",
        "resnet50",
        128,
        val_fooling_rate_clipped=0.42,
        val_random_baseline_clipped=0.17,
    )

    run = load_sweep(str(tmp_path / "images"), "images")[0]

    assert run["margin"] == pytest.approx(0.25)


def test_margin_is_negative_when_worse_than_noise(tmp_path):
    """The case the whole study turns on must survive the export."""
    _summary(
        tmp_path,
        "images",
        "inception5h",
        16,
        val_fooling_rate_clipped=0.12,
        val_random_baseline_clipped=0.325,
    )

    run = load_sweep(str(tmp_path / "images"), "images")[0]

    assert run["margin"] < 0
    assert run["margin"] == pytest.approx(-0.205)


def test_runs_carry_a_paradigm_label(tmp_path):
    _summary(tmp_path, "modern", "vit_b16", 128)
    run = load_sweep(str(tmp_path / "modern"), "modern")[0]
    assert "transformer" in run["paradigm"]


def test_unknown_models_do_not_break_the_export(tmp_path):
    _summary(tmp_path, "images", "some_new_model", 128)
    run = load_sweep(str(tmp_path / "images"), "images")[0]
    assert run["paradigm"] == "unknown"


def test_missing_directory_yields_nothing(tmp_path):
    assert load_sweep(str(tmp_path / "nope"), "images") == []


def test_camel_case_keys_match_what_the_site_reads(tmp_path):
    _summary(tmp_path, "images", "resnet50", 128)
    run = load_sweep(str(tmp_path / "images"), "images")[0]

    for key in (
        "numImages",
        "numValImages",
        "searchNum",
        "genFooling",
        "valFooling",
        "randomBaseline",
        "margin",
        "cleanTop1",
    ):
        assert key in run, key


def test_every_registered_model_has_a_paradigm():
    from classifiers import AVAILABLE_MODELS

    missing = [m for m in AVAILABLE_MODELS if m not in PARADIGMS]
    assert not missing, f"models without a paradigm label: {missing}"
