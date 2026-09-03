# Universal adversarial perturbations for fast and high error rate

[![CI](https://github.com/Uno-Takashi/Universal-Adversarial-Perturbation-for-faster-and-higher-error-rate/actions/workflows/ci.yml/badge.svg)](https://github.com/Uno-Takashi/Universal-Adversarial-Perturbation-for-faster-and-higher-error-rate/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

*[日本語版 README](README.ja.md)*

This repo extends [Universal Adversarial Perturbation](https://github.com/LTS4/universal) to reach a
high error rate quickly, from a small number of images.

A single image-agnostic perturbation `v` is computed such that adding it to most images of a dataset
changes the classifier's prediction. Where the original algorithm walks each image to its nearest
decision boundary, this one targets the top `M` runner-up classes per image — the multiplicity
parameter, `search_num` in the code — which reaches a higher fooling rate from fewer images.

## Requirements

- Python 3.11 – 3.13
- [uv](https://docs.astral.sh/uv/) for dependency management
- An NVIDIA GPU is optional; TensorFlow falls back to the CPU

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
# CPU, or any machine that already has CUDA and cuDNN installed system-wide
uv sync

# NVIDIA GPU without a system CUDA install: pulls the CUDA libraries as wheels
uv sync --extra cuda
```

`uv sync` creates `.venv` and installs the exact versions from `uv.lock`. Prefix commands with
`uv run` to use it, or activate it with `source .venv/bin/activate`.

> [!IMPORTANT]
> If you use the `cuda` extra, pass it to `uv run` as well (`uv run --extra cuda python ...`) or set
> `UV_NO_SYNC=1`. A bare `uv run` re-syncs the environment to the default extras and uninstalls the
> CUDA wheels. This does not apply inside the dev container, which gets CUDA from its base image and
> so needs no extra.

### Dev container

A [dev container](https://containers.dev/) is provided in [`.devcontainer/`](.devcontainer/). It is
built on `nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04` and declares
`"hostRequirements": { "gpu": "optional" }`, so it uses the GPU when the host exposes one and runs on
the CPU otherwise. "Reopen in Container" from VS Code runs `uv sync --group dev` and reports whether
TensorFlow can see a GPU.

The base image supplies CUDA 12.6 and cuDNN system-wide, so the plain `tensorflow` wheel picks them
up; the `cuda` extra is only for GPU machines *outside* the container.

GPU passthrough additionally requires, on the host, a recent NVIDIA driver, the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
and Docker configured with the `nvidia` runtime. On WSL2 install the driver on Windows (not inside
WSL) and enable GPU support in Docker Desktop.

## Usage

### Demo: apply a perturbation to one image

```bash
uv run python demo_inception.py -i data/test_img.png
```

This downloads the Inception 5h model and shows the image with and without the universal
perturbation, labelled with the estimated classes. Add `-o figure.png` to write the figure to a file
instead of opening a window.

> [!NOTE]
> `data/universal.npy` is not committed to this repository. When it is absent, `demo_inception.py`
> computes a perturbation first, streaming images from Hugging Face (`-n` controls how many). Pass
> `-t /path/to/ILSVRC2012/train` to use a local ImageNet tree instead.

### Experiments across models

[`experiment.py`](experiment.py) runs the algorithm end to end and reports fooling rates. Everything
it needs downloads itself — model weights and ILSVRC images — so a run reproduces on a fresh machine
with nothing but `uv sync`.

```bash
# One model
uv run python experiment.py --models inception5h --num-images 100

# Does the algorithm generalise past Inception?
uv run python experiment.py --models inception5h mobilenet_v2 resnet50 --num-images 100 --transfer

# Sweep the multiplicity M
uv run python experiment.py --models mobilenet_v2 --search-num 1 3 5 10

# Use a real local ILSVRC tree instead of Hugging Face
uv run python experiment.py --dataset /datasets2/ILSVRC2012/train --num-images 500
```

It prints a table per run and, with `--out DIR`, saves each perturbation as `.npy` alongside a
`summary.json`:

```
model               M  images   clean    fool  fool(clip)    sec
----------------------------------------------------------------
inception5h         3       8   75.0%   87.5%       87.5%      7
mobilenet_v2        3       8   72.0%  100.0%      100.0%      7
resnet50            3       8   87.5%  100.0%      100.0%     18
```

`fool` is the fraction of images whose predicted label changes — the criterion the algorithm
optimises. `fool(clip)` is the same measure after clipping the perturbed image back into
`[0, 255]`, i.e. for images that are still valid pictures. `--transfer` additionally evaluates every
perturbation against the other models of matching input size.

### Supported models

| Name | Source | Input | Classes |
|---|---|---|---|
| `inception5h` | frozen graph (the paper's model) | 224×224 | 1008 |
| `inception_v3` | `tf.keras.applications` | 299×299 | 1000 |
| `resnet50` | `tf.keras.applications` | 224×224 | 1000 |
| `mobilenet_v2` | `tf.keras.applications` | 224×224 | 1000 |
| `vgg16` | `tf.keras.applications` | 224×224 | 1000 |
| `efficientnet_b0` | `tf.keras.applications` | 224×224 | 1000 |
| `convnext_tiny` | `tf.keras.applications` | 224×224 | 1000 |

To add your own, subclass [`classifiers.Classifier`](classifiers.py) and implement `_forward`
(raw pixels to pre-softmax logits) and `label`. Gradients come for free from `tf.GradientTape`.

### Datasets

Images come from the Hugging Face dataset viewer, so only the images you ask for are fetched — a
50-image run downloads 50 images, not a multi-hundred-megabyte shard.

| Shorthand | Dataset | Note |
|---|---|---|
| `imagenet-val` (default) | `NightMachinery/ImageNet1K-val-indexed` | ILSVRC2012 validation, ungated |
| `imagenet-val-alt` | `danaroth/imagenet_val` | ungated mirror |
| `imagenet-val-224` | `danjacobellis/imagenet_1k_val_224` | pre-cropped to 224 |
| `imagenet-1k` | `imagenet-1k` | gated: set `HF_TOKEN` and accept the terms |

Any hub id works in place of a shorthand, as does a path to a local ILSVRC-style directory (one
subdirectory per class).

### Computing a perturbation in your own code

```python
from classifiers import build_classifier
from imagenet_source import load_images
from universal_pert import universal_perturbation

model = build_classifier("resnet50")
dataset, _ = load_images("imagenet-val", num_images=50, image_size=model.image_size)

v = universal_perturbation(dataset, model, model.gradients, delta=0.2, search_num=5)
```

`universal_perturbation` takes a feedforward callable and a gradient callable, so it works with any
model you can differentiate — see [`universal_pert.py`](universal_pert.py) for the parameters.

## Design notes

**Raw pixel space.** Every classifier accepts raw RGB in `[0, 255]` and applies its own
preprocessing inside the TensorFlow graph. Perturbations and the `xi` budget are then expressed in
pixel levels, so the same number means the same thing across models, and clipping is a plain
`clip(raw + v, 0, 255)`. For Inception 5h this is equivalent to the original mean-subtracted
formulation: subtracting a constant is a shift, so it leaves gradients and l_p norms untouched, and
`data/universal.npy` stays valid.

**TF2 native.** The model is an eager callable and gradients come from `tf.GradientTape`; there is
no session, placeholder or feed dict, and no hand-rolled `tf.while_loop` Jacobian.
`tf.compat.v1.wrap_function` in [`classifiers.py`](classifiers.py) is the single remaining compat
call — a frozen `GraphDef` has no other TF2 entry point — and
[`tests/test_tf2_native.py`](tests/test_tf2_native.py) enforces that it stays the only one.

**Module layout.**

| Module | Role |
|---|---|
| [`deepfool.py`](deepfool.py) | untargeted per-image attack |
| [`deeptarget.py`](deeptarget.py) | targeted per-image attack, the inner loop of this variant |
| [`universal_pert.py`](universal_pert.py) | the universal perturbation algorithm |
| [`classifiers.py`](classifiers.py) | model abstraction and implementations |
| [`imagenet_source.py`](imagenet_source.py) | image loading from Hugging Face or disk |
| [`experiment.py`](experiment.py) | experiment driver and reporting |
| [`util_univ.py`](util_univ.py) | fooling-rate metrics and clipping helpers |
| [`prepare_imagenet_data.py`](prepare_imagenet_data.py) | legacy mean-subtraction preprocessing helpers |

The algorithm modules are pure NumPy and take callables, so they carry no TensorFlow dependency.

## Development

```bash
uv sync --group dev        # install the dev tooling
uv run ruff check .        # lint
uv run ruff format .       # format
uv run pytest              # run the test suite
```

Linting, formatting, the test suite (Python 3.11/3.12/3.13), the lockfile, and the package build all
run on every push and pull request against `main`; see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml). The tests need no network and no model
weights: the algorithms are checked against a synthetic linear classifier with an analytic Jacobian,
and the Hugging Face loader against a fake HTTP layer.

## Releasing

Releases are cut from a tag. Bump `version` in `pyproject.toml`, then:

```bash
git tag v0.2.0
git push origin v0.2.0
```

[`.github/workflows/release.yml`](.github/workflows/release.yml) verifies that the tag matches the
project version, runs the tests, builds the sdist and wheel, and publishes a GitHub Release with
those artifacts attached. Publishing to PyPI is opt-in: set the repository variable
`PUBLISH_TO_PYPI` to `true` and configure PyPI trusted publishing for the `pypi` environment.

## Reference

[1] S. Moosavi-Dezfooli\*, A. Fawzi\*, O. Fawzi, P. Frossard:
[*Universal adversarial perturbations*](http://arxiv.org/pdf/1610.08401), CVPR 2017.
