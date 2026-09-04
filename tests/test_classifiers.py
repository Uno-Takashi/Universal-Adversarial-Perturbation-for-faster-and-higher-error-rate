"""Tests for the classifier abstraction, using a tiny in-process model.

A real ImageNet model would have to be downloaded, so the interface is exercised through a
linear stand-in. The Inception 5h and Keras subclasses are only checked for the parts that
need no weights.
"""

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf

from classifiers import (
    AVAILABLE_MODELS,
    KERAS_HUB_MODELS,
    KERAS_MODELS,
    Classifier,
    Inception5hClassifier,
    KerasClassifier,
    KerasHubClassifier,
    _normalize_label,
    build_classifier,
)

NUM_CLASSES = 6
IMAGE_SIZE = (8, 8)


class LinearClassifier(Classifier):
    """logits = W @ flatten(raw / 255), so gradients are constant and easy to predict."""

    name = "linear"
    image_size = IMAGE_SIZE
    num_classes = NUM_CLASSES

    def __init__(self, seed=0):
        n_features = int(np.prod(IMAGE_SIZE)) * 3
        rng = np.random.default_rng(seed)
        self.weights = tf.constant(rng.normal(size=(n_features, NUM_CLASSES)), dtype=tf.float32)

    def _forward(self, raw_batch):
        flat = tf.reshape(raw_batch, (tf.shape(raw_batch)[0], -1)) / 255.0
        return tf.matmul(flat, self.weights)

    def label(self, index):
        return f"class_{int(np.ravel(index)[0])}"


@pytest.fixture
def model():
    return LinearClassifier()


@pytest.fixture
def batch():
    rng = np.random.default_rng(1)
    return rng.uniform(0, 255, size=(3, *IMAGE_SIZE, 3)).astype(np.float32)


def test_call_returns_logits_per_image(model, batch):
    logits = model(batch)
    assert logits.shape == (3, NUM_CLASSES)
    assert np.all(np.isfinite(logits))


def test_call_accepts_a_single_unbatched_image(model):
    single = np.zeros((*IMAGE_SIZE, 3), dtype=np.float32)
    assert model(single).shape == (1, NUM_CLASSES)


def test_predict_matches_argmax_of_logits(model, batch):
    np.testing.assert_array_equal(model.predict(batch), np.argmax(model(batch), axis=1))


def test_gradients_have_the_shape_the_algorithms_index_into(model, batch):
    indices = [0, 2, 5]
    grads = model.gradients(batch[:1], indices)
    assert grads.shape == (len(indices), 1, *IMAGE_SIZE, 3)
    assert np.all(np.isfinite(grads))


def test_gradients_match_the_analytic_jacobian(model, batch):
    """For this linear model d(logit_c)/d(raw) is W[:, c] / 255, reshaped."""
    indices = [1, 4]
    grads = model.gradients(batch[:1], indices)

    weights = model.weights.numpy() / 255.0
    for position, class_index in enumerate(indices):
        expected = weights[:, class_index].reshape(1, *IMAGE_SIZE, 3)
        np.testing.assert_allclose(grads[position], expected, atol=1e-5)


def test_gradients_accept_a_varying_number_of_indices(model, batch):
    assert model.gradients(batch[:1], [0]).shape[0] == 1
    assert model.gradients(batch[:1], [0, 1, 2, 3]).shape[0] == 4


def test_labels_of_returns_one_label_per_image(model, batch):
    labels = model.labels_of(batch)
    assert len(labels) == 3
    assert all(label.startswith("class_") for label in labels)


def test_to_ilsvrc_is_the_identity_by_default(model):
    np.testing.assert_array_equal(model.to_ilsvrc([3, 7]), np.array([3, 7]))


def test_classifier_is_abstract():
    with pytest.raises(TypeError):
        Classifier()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("kit fox, Vulpes macrotis", "kit_fox_vulpes_macrotis"),
        ("  Egyptian cat ", "egyptian_cat"),
        ("bell pepper", "bell_pepper"),
        ("n01440764", "n01440764"),
    ],
)
def test_normalize_label(raw, expected):
    assert _normalize_label(raw) == expected


def test_available_models_covers_inception_and_the_keras_zoo():
    assert "inception5h" in AVAILABLE_MODELS
    for name in KERAS_MODELS:
        assert name in AVAILABLE_MODELS


def test_keras_models_declare_plausible_input_sizes():
    assert KERAS_MODELS["inception_v3"][2] == (299, 299)
    for _, _, size in KERAS_MODELS.values():
        assert size[0] == size[1]
        assert 96 <= size[0] <= 600


def test_build_classifier_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown Keras model"):
        build_classifier("definitely-not-a-model")


def test_keras_classifier_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown Keras model"):
        KerasClassifier("nope")


def test_inception_label_lookup_uses_the_one_lower_offset(tmp_path):
    labels = tmp_path / "labels.txt"
    labels.write_text("kit fox, Vulpes macrotis\nEnglish setter\nSiberian husky\n")

    # __init__ would load the graph, so drive the label path on a bare instance.
    model = Inception5hClassifier.__new__(Inception5hClassifier)
    model._labels_path = str(labels)
    model._labels = None

    assert model.label(1) == "kit fox"
    assert model.label(2) == "English setter"
    assert model.label([3]) == "Siberian husky"


def test_inception_label_lookup_survives_out_of_range_ids(tmp_path):
    labels = tmp_path / "labels.txt"
    labels.write_text("only one\n")

    model = Inception5hClassifier.__new__(Inception5hClassifier)
    model._labels_path = str(labels)
    model._labels = None

    assert model.label(0) == "class_0"
    assert model.label(500) == "class_500"


def test_keras_hub_models_cover_the_post_cnn_paradigms():
    """The point of the KerasHub entries is architectural coverage, not more CNNs."""
    paradigms = {paradigm for _, paradigm in KERAS_HUB_MODELS.values()}

    assert "vision transformer" in paradigms
    assert "hierarchical windowed transformer" in paradigms
    assert any("distillation" in p for p in paradigms)


def test_keras_hub_presets_are_imagenet_classifiers():
    for name, (preset, _) in KERAS_HUB_MODELS.items():
        assert isinstance(preset, str) and preset, name
        assert "imagenet" in preset or "224" in preset, (name, preset)


def test_keras_hub_classifier_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown KerasHub model"):
        KerasHubClassifier("not-a-real-model")


def test_available_models_lists_every_registry():
    for name in KERAS_MODELS:
        assert name in AVAILABLE_MODELS
    for name in KERAS_HUB_MODELS:
        assert name in AVAILABLE_MODELS
    assert len(set(AVAILABLE_MODELS)) == len(AVAILABLE_MODELS), "duplicate model name"


def test_build_classifier_routes_hub_names_to_the_hub_class(monkeypatch):
    built = {}

    def fake_init(self, name, preset=None):
        built["name"] = name

    monkeypatch.setattr(KerasHubClassifier, "__init__", fake_init)
    result = build_classifier("vit_b16")

    assert isinstance(result, KerasHubClassifier)
    assert built["name"] == "vit_b16"


def test_build_classifier_still_routes_applications_names(monkeypatch):
    built = {}

    def fake_init(self, name, weights="imagenet"):
        built["name"] = name

    monkeypatch.setattr(KerasClassifier, "__init__", fake_init)
    result = build_classifier("resnet50")

    assert isinstance(result, KerasClassifier)
    assert built["name"] == "resnet50"
