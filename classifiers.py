"""Classifiers the perturbation algorithms can attack, behind one small interface.

Every classifier takes **raw RGB pixels in [0, 255]** shaped ``(N, H, W, 3)`` and applies its
own preprocessing inside the TensorFlow graph. That matters for two reasons:

* Perturbations, and the l_p radius ``xi`` that bounds them, are then expressed in pixel
  levels, so the same number means the same thing for every model. Model-specific
  preprocessing (mean subtraction, ``[-1, 1]`` scaling, ...) would otherwise rescale it.
* Clipping a perturbed image back into the displayable range is a plain
  ``clip(raw + v, 0, 255)``, with no per-model inverse to get wrong.

For Inception 5h this is equivalent to the original mean-subtracted formulation: subtracting a
constant is a shift, so it leaves gradients and l_p norms untouched, and a perturbation
computed either way is the same array.

Gradients come from :class:`tf.GradientTape`, so nothing here needs a session, a placeholder
or a hand-rolled ``tf.while_loop``.
"""

import json
import os
import re
import zipfile
from abc import ABC, abstractmethod
from urllib.request import urlretrieve

import numpy as np

from gpu_support import enable_cuda_wheels

# Must precede `import tensorflow`: TensorFlow resolves its CUDA libraries during import.
enable_cuda_wheels()

import tensorflow as tf  # noqa: E402

INCEPTION_URL = "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
INCEPTION_GRAPH = "tensorflow_inception_graph.pb"
INCEPTION_INPUT = "input:0"
INCEPTION_OUTPUT = "softmax2_pre_activation:0"

# Empirical channel means of the ILSVRC2012 training set (RGB order), as used by Inception 5h.
CHANNEL_MEANS = (123.68, 116.779, 103.939)

KERAS_CLASS_INDEX_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
)


def _normalize_label(name):
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def ilsvrc_class_names():
    """Return the 1000 ILSVRC class names in the standard synset order, normalised."""
    global _ILSVRC_NAMES
    if _ILSVRC_NAMES is None:
        path = tf.keras.utils.get_file("imagenet_class_index.json", KERAS_CLASS_INDEX_URL)
        with open(path) as fh:
            mapping = json.load(fh)
        _ILSVRC_NAMES = [_normalize_label(mapping[str(i)][1]) for i in range(len(mapping))]
    return _ILSVRC_NAMES


_ILSVRC_NAMES = None


def download_inception_graph(data_dir="data"):
    """Download and unpack the Inception 5h frozen graph into ``data_dir``."""
    graph_path = os.path.join(data_dir, INCEPTION_GRAPH)
    if os.path.isfile(graph_path):
        return graph_path

    os.makedirs(data_dir, exist_ok=True)
    archive = os.path.join(data_dir, "inception5h.zip")
    print("Downloading Inception 5h model...")
    urlretrieve(INCEPTION_URL, archive)
    with zipfile.ZipFile(archive, "r") as zip_ref:
        zip_ref.extract(INCEPTION_GRAPH, data_dir)
    return graph_path


def load_frozen_graph(graph_path, input_name=INCEPTION_INPUT, output_name=INCEPTION_OUTPUT):
    """Import a frozen ``GraphDef`` and return it as an eager-callable concrete function.

    This is the migration path TensorFlow documents for frozen graphs. The
    ``tf.compat.v1.wrap_function`` call is the only compat shim in the project: a frozen
    ``GraphDef`` has no other TF2 entry point. What it returns is an ordinary TF2 function.
    """
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(graph_path, "rb") as fh:
        graph_def.ParseFromString(fh.read())

    def _import_graph():
        tf.graph_util.import_graph_def(graph_def, name="")

    wrapped = tf.compat.v1.wrap_function(_import_graph, [])
    return wrapped.prune(
        tf.nest.map_structure(wrapped.graph.as_graph_element, input_name),
        tf.nest.map_structure(wrapped.graph.as_graph_element, output_name),
    )


class Classifier(ABC):
    """A differentiable image classifier over raw ``[0, 255]`` RGB input.

    Instances are callable, so a classifier can be passed straight in as the ``f`` argument
    of :func:`universal_pert.universal_perturbation`, with :meth:`gradients` as ``grads``.
    """

    name = "classifier"
    image_size = (224, 224)
    num_classes = 1000

    @abstractmethod
    def _forward(self, raw_batch):
        """Map a raw ``[0, 255]`` float32 tensor to pre-softmax logits."""

    @abstractmethod
    def label(self, index):
        """Return the human-readable label for a class index."""

    def _as_batch(self, images):
        array = np.reshape(np.asarray(images, dtype=np.float32), (-1, *self.image_size, 3))
        return tf.convert_to_tensor(array)

    def __call__(self, images):
        """Return pre-softmax activations shaped ``(N, num_classes)``."""
        return np.asarray(self._forward_batched(self._as_batch(images)))

    def gradients(self, images, indices):
        """Return ``d(logit[i])/d(raw input)`` for every ``i`` in ``indices``.

        Shaped ``(len(indices), N, H, W, 3)``, which is what :func:`deepfool.deepfool` and
        :func:`deeptarget.deeptarget` index into.
        """
        batch = self._as_batch(images)
        inds = tf.convert_to_tensor(np.asarray(indices, dtype=np.int32))
        return np.asarray(self._jacobian(batch, inds))

    def predict(self, images):
        """Return the predicted class index for each image."""
        return np.argmax(self(images), axis=1).flatten()

    def to_ilsvrc(self, indices):
        """Map this model's class indices onto the standard ILSVRC 0-999 ordering.

        Returns -1 for indices with no ILSVRC counterpart. Models that already use the
        standard ordering need no translation.
        """
        return np.asarray(indices, dtype=np.int64)

    def labels_of(self, images):
        return [self.label(i) for i in self.predict(images)]

    @tf.function(reduce_retracing=True)
    def _forward_batched(self, batch):
        return self._forward(batch)

    @tf.function(reduce_retracing=True)
    def _jacobian(self, batch, inds):
        with tf.GradientTape() as tape:
            tape.watch(batch)
            logits = tf.reshape(self._forward(batch), (-1,))
            selected = tf.gather(logits, inds)
        return tape.jacobian(selected, batch)


class Inception5hClassifier(Classifier):
    """Inception 5h, loaded from its frozen graph.

    This is the model the original paper and this repository's ``data/universal.npy`` use. Its
    ``softmax2`` head has 1008 outputs, and ``data/labels.txt`` is indexed one lower than the
    class id -- both quirks of the released graph, preserved here.
    """

    name = "inception5h"
    image_size = (224, 224)
    num_classes = 1008

    def __init__(self, graph_path=None, data_dir="data", labels_path=None):
        self._fn = load_frozen_graph(graph_path or download_inception_graph(data_dir))
        self._labels_path = labels_path or os.path.join(data_dir, "labels.txt")
        self._labels = None
        self._ilsvrc_table = None
        self._means = tf.constant(CHANNEL_MEANS, dtype=tf.float32)

    def _forward(self, raw_batch):
        return self._fn(raw_batch - self._means)

    def _label_names(self):
        if self._labels is None:
            with open(self._labels_path) as fh:
                # The file carries 1000 names (plus a trailing blank from the final newline)
                # and is indexed one lower than the class id of the 1008-way softmax2 head.
                self._labels = fh.read().split("\n")
        return self._labels

    def label(self, index):
        names = self._label_names()
        position = int(np.ravel(index)[0]) - 1
        if not 0 <= position < len(names):
            return f"class_{int(np.ravel(index)[0])}"
        return names[position].split(",")[0]

    def to_ilsvrc(self, indices):
        """Translate Inception 5h class ids to ILSVRC ids by label name.

        Inception 5h ships its own label ordering: all 1000 ILSVRC names are present but
        permuted, so an index offset does not line the two up -- only the names do.
        """
        if self._ilsvrc_table is None:
            ilsvrc = {name: i for i, name in enumerate(ilsvrc_class_names())}
            table = np.full(self.num_classes, -1, dtype=np.int64)
            for class_id in range(self.num_classes):
                name = _normalize_label(self.label(class_id))
                table[class_id] = ilsvrc.get(name, -1)
            self._ilsvrc_table = table
        return self._ilsvrc_table[np.asarray(indices, dtype=np.int64)]


# name -> (keras class attribute, preprocessing module attribute, input size)
KERAS_MODELS = {
    "inception_v3": ("InceptionV3", "inception_v3", (299, 299)),
    "resnet50": ("ResNet50", "resnet50", (224, 224)),
    "mobilenet_v2": ("MobileNetV2", "mobilenet_v2", (224, 224)),
    "vgg16": ("VGG16", "vgg16", (224, 224)),
    "efficientnet_b0": ("EfficientNetB0", "efficientnet", (224, 224)),
    "convnext_tiny": ("ConvNeXtTiny", "convnext", (224, 224)),
}


class KerasClassifier(Classifier):
    """Any ImageNet model from ``tf.keras.applications``.

    Weights download on first use. ``classifier_activation=None`` is required: DeepFool needs
    pre-softmax logits, and Keras applies a softmax head by default.
    """

    num_classes = 1000

    def __init__(self, name, weights="imagenet"):
        if name not in KERAS_MODELS:
            raise ValueError(f"Unknown Keras model {name!r}. Available: {sorted(KERAS_MODELS)}")
        import tensorflow.keras.applications as apps

        class_name, preprocess_module, image_size = KERAS_MODELS[name]
        self.name = name
        self.image_size = image_size
        self._model = getattr(apps, class_name)(weights=weights, classifier_activation=None)
        self._preprocess = getattr(apps, preprocess_module).preprocess_input
        self._labels = None

    def _forward(self, raw_batch):
        return self._model(self._preprocess(raw_batch), training=False)

    def label(self, index):
        if self._labels is None:
            path = tf.keras.utils.get_file("imagenet_class_index.json", KERAS_CLASS_INDEX_URL)
            with open(path) as fh:
                mapping = json.load(fh)
            self._labels = [mapping[str(i)][1] for i in range(len(mapping))]
        return self._labels[int(np.ravel(index)[0])]


def build_classifier(name, data_dir="data"):
    """Build a classifier by name: ``inception5h`` or any key of :data:`KERAS_MODELS`."""
    if name == "inception5h":
        return Inception5hClassifier(data_dir=data_dir)
    return KerasClassifier(name)


AVAILABLE_MODELS = ("inception5h", *KERAS_MODELS)
