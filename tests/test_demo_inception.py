"""Smoke checks for the demo entrypoint.

Only import-time and argument-parsing behaviour is covered; running the demo needs the
Inception graph and a test image, which are out of scope for CI.
"""

import pytest

pytest.importorskip("tensorflow", reason="TensorFlow is required for the demo module")

import demo_inception


def test_module_exposes_its_entrypoints():
    assert callable(demo_inception.main)
    assert callable(demo_inception.parse_args)
    assert callable(demo_inception.load_image)
    assert demo_inception.NUM_CLASSES == 2


def test_tensorflow_runs_eagerly():
    import tensorflow as tf

    assert tf.executing_eagerly()


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["demo_inception.py"])
    args = demo_inception.parse_args()

    assert args.test_image.endswith("test_img.png")
    # None means "stream from Hugging Face" rather than a hard-coded local path.
    assert args.training_path is None
    assert args.num_images == 50
    assert args.output is None


def test_parse_args_short_flags(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["demo_inception.py", "-i", "a.png", "-t", "/data/train", "-n", "7", "-o", "fig.png"],
    )
    args = demo_inception.parse_args()

    assert args.test_image == "a.png"
    assert args.training_path == "/data/train"
    assert args.num_images == 7
    assert args.output == "fig.png"
