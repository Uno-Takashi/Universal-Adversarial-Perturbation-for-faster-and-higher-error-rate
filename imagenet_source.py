"""Load ImageNet images as raw ``[0, 255]`` RGB batches, from Hugging Face or a local tree.

Hugging Face images arrive through the dataset-viewer ``/rows`` endpoint rather than the
``datasets`` library. That keeps an experiment cheap and portable: a run on 50 images fetches
50 images, not a multi-hundred-megabyte shard, it needs nothing beyond the standard library
and Pillow, and it avoids a crash on interpreter shutdown that ``datasets`` streaming
triggers in this environment (reproduced on 3.6, 4.4 and 5.0).

Both loaders return the same thing: an ``(N, H, W, 3)`` float32 array of raw pixel values plus
the ground-truth labels, ready to hand to any :class:`classifiers.Classifier`.
"""

import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
MAX_ROWS_PER_REQUEST = 100
# Statuses worth retrying: viewer cache warm-up and rate limiting.
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

# Ungated mirrors of the ILSVRC2012 validation split, already shuffled across classes.
# `imagenet-1k` itself is gated: set HF_TOKEN and accept the dataset terms to use it.
KNOWN_DATASETS = {
    "imagenet-val": ("NightMachinery/ImageNet1K-val-indexed", "train"),
    "imagenet-val-alt": ("danaroth/imagenet_val", "validation"),
    "imagenet-val-224": ("danjacobellis/imagenet_1k_val_224", "validation"),
    "imagenet-1k": ("imagenet-1k", "validation"),
}
DEFAULT_DATASET = "imagenet-val"

_IMAGE_COLUMNS = ("image", "img", "crop224", "jpg", "png")
_LABEL_COLUMNS = ("label", "cls", "fine_label", "labels", "class")


def resize_and_crop(image, image_size, resize_ratio=256 / 224):
    """Resize, then centre-crop to ``image_size``.

    ``resize_ratio`` reproduces the original resize-to-256 / crop-to-224 recipe and scales it
    to whatever input size a model wants (299 becomes resize 342, crop 299).
    """
    target_h, target_w = image_size
    resize_h = round(target_h * resize_ratio)
    resize_w = round(target_w * resize_ratio)

    image = image.convert("RGB").resize((resize_w, resize_h), Image.BILINEAR)
    left = (resize_w - target_w) // 2
    top = (resize_h - target_h) // 2
    return image.crop((left, top, left + target_w, top + target_h))


def _auth_headers():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _fetch(url, attempts=5, backoff=2.0):
    """GET ``url``, retrying transient failures.

    The dataset viewer returns 502/503 while it warms a cache up, and a long run makes
    thousands of requests, so a retry here is the difference between an experiment that
    finishes and one that dies halfway.
    """
    last_error = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=_auth_headers())
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            status = getattr(error, "code", None)
            if status is not None and status not in _RETRY_STATUS:
                raise
            last_error = error
            if attempt < attempts - 1:
                time.sleep(backoff * (2**attempt))
    raise RuntimeError(f"Gave up on {url} after {attempts} attempts: {last_error}")


def _get_json(url):
    return json.loads(_fetch(url))


def _get_bytes(url):
    return _fetch(url)


def _rows_url(dataset, config, split, offset, length):
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    return f"{ROWS_ENDPOINT}?{query}"


def _pick_column(row, candidates, kind):
    for name in candidates:
        if name in row:
            return name
    if kind == "image":
        # Hugging Face serves images as {"src": url} or as raw bytes; find either.
        for name, value in row.items():
            if isinstance(value, dict) and "src" in value:
                return name
    return None


def _decode_image(value):
    if isinstance(value, dict) and "src" in value:
        return Image.open(io.BytesIO(_get_bytes(value["src"])))
    if isinstance(value, dict) and "bytes" in value:
        return Image.open(io.BytesIO(value["bytes"]))
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value))
    if isinstance(value, str):
        return Image.open(io.BytesIO(_get_bytes(value)))
    raise TypeError(f"Cannot decode an image from {type(value).__name__}")


def load_hf_images(
    dataset=DEFAULT_DATASET,
    split=None,
    num_images=50,
    image_size=(224, 224),
    seed=0,
    config="default",
):
    """Fetch ``num_images`` images from a Hugging Face dataset.

    ``dataset`` takes a shorthand from :data:`KNOWN_DATASETS` or any hub id. A random
    contiguous window is read, which is a class-diverse sample because the mirrors are
    already shuffled. Pass ``seed=None`` to always start at offset 0.

    Returns ``(images, labels)``; ``labels`` is empty when the dataset exposes no label
    column.
    """
    if dataset in KNOWN_DATASETS:
        dataset, default_split = KNOWN_DATASETS[dataset]
    else:
        default_split = "train"
    split = split or default_split

    probe = _get_json(_rows_url(dataset, config, split, 0, 1))
    total = int(probe.get("num_rows_total") or 0)
    if not probe.get("rows"):
        raise ValueError(f"{dataset!r} split {split!r} returned no rows")
    if total and num_images > total:
        raise ValueError(f"{dataset!r} has only {total} rows; asked for {num_images}")

    start = 0
    if seed is not None and total > num_images:
        start = int(np.random.default_rng(seed).integers(0, total - num_images + 1))

    first_row = probe["rows"][0]["row"]
    image_column = _pick_column(first_row, _IMAGE_COLUMNS, "image")
    label_column = _pick_column(first_row, _LABEL_COLUMNS, "label")
    if image_column is None:
        raise ValueError(f"Could not find an image column in {dataset!r}; saw {sorted(first_row)}")

    images = []
    labels = []
    offset = start
    while len(images) < num_images:
        length = min(MAX_ROWS_PER_REQUEST, num_images - len(images))
        payload = _get_json(_rows_url(dataset, config, split, offset, length))
        rows = payload.get("rows") or []
        if not rows:
            break
        for entry in rows:
            row = entry["row"]
            with _decode_image(row[image_column]) as handle:
                images.append(np.asarray(resize_and_crop(handle, image_size), dtype=np.float32))
            if label_column is not None:
                labels.append(int(row[label_column]))
        offset += len(rows)

    if not images:
        raise ValueError(f"No images loaded from {dataset!r} split {split!r}")

    return np.stack(images, axis=0), np.asarray(labels, dtype=np.int64)


def load_local_images(root, num_images=50, image_size=(224, 224), per_class=None, seed=0):
    """Load images from an ILSVRC-style directory tree (one subdirectory per class).

    Images are taken from the class directories in sorted order, ``per_class`` at a time, so
    the sample spreads across classes the way the original ``create_imagenet_npy`` did. When
    ``per_class`` is None it is derived from ``num_images`` and the number of classes.
    """
    class_dirs = sorted(
        os.path.join(root, name)
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    )
    if not class_dirs:
        raise ValueError(f"No class subdirectories found under {root!r}")

    if per_class is None:
        per_class = max(1, int(np.ceil(num_images / len(class_dirs))))

    images = []
    for class_dir in class_dirs:
        for name in sorted(os.listdir(class_dir))[:per_class]:
            try:
                with Image.open(os.path.join(class_dir, name)) as handle:
                    images.append(np.asarray(resize_and_crop(handle, image_size), dtype=np.float32))
            except OSError:
                continue
            if len(images) >= num_images:
                break
        if len(images) >= num_images:
            break

    if not images:
        raise ValueError(f"No readable images found under {root!r}")

    stacked = np.stack(images, axis=0)
    if len(stacked) > num_images:
        keep = np.random.default_rng(seed).permutation(len(stacked))[:num_images]
        stacked = stacked[keep]
    return stacked, np.asarray([], dtype=np.int64)


def load_images(source=DEFAULT_DATASET, num_images=50, image_size=(224, 224), split=None, seed=0):
    """Load from a local directory when ``source`` is a path, else from Hugging Face."""
    if source and os.path.isdir(source):
        return load_local_images(source, num_images=num_images, image_size=image_size, seed=seed)
    return load_hf_images(
        source, split=split, num_images=num_images, image_size=image_size, seed=seed
    )
