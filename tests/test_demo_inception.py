"""Smoke checks for the TensorFlow-backed demo entrypoint.

Only import-time and argument-parsing behaviour is covered here; running the demo needs the
Inception graph and a test image, which are out of scope for CI.
"""

import pytest

pytest.importorskip("tensorflow", reason="TensorFlow is required for the demo module")

import demo_inception


def test_module_exposes_its_entrypoints():
    assert callable(demo_inception.main)
    assert callable(demo_inception.jacobian)
    assert demo_inception.NUM_CLASSES == 2


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["demo_inception.py"])
    args = demo_inception.parse_args()

    assert args.test_image.endswith("test_img.png")
    assert args.training_path == "/datasets2/ILSVRC2012/train"


def test_parse_args_short_flags(monkeypatch):
    monkeypatch.setattr("sys.argv", ["demo_inception.py", "-i", "a.png", "-t", "/data/train"])
    args = demo_inception.parse_args()

    assert args.test_image == "a.png"
    assert args.training_path == "/data/train"
