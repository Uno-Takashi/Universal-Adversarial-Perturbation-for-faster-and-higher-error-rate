# Universal adversarial perturbations for fast and high error rate

[![CI](https://github.com/Uno-Takashi/Universal-Adversarial-Perturbation-for-faster-and-higher-error-rate/actions/workflows/ci.yml/badge.svg)](https://github.com/Uno-Takashi/Universal-Adversarial-Perturbation-for-faster-and-higher-error-rate/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

This repo extends [Universal Adversarial Perturbation](https://github.com/LTS4/universal) to reach a high error rate quickly from a small amount of data.

Python code to generate universal perturbations using [TensorFlow](https://github.com/tensorflow/tensorflow). Dependencies are declared in `pyproject.toml` and locked in `uv.lock`.

## Requirements

- Python 3.11 - 3.13
- [uv](https://docs.astral.sh/uv/) for dependency management
- An NVIDIA GPU is optional; TensorFlow falls back to the CPU

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
# CPU
uv sync

# NVIDIA GPU (installs tensorflow[and-cuda])
uv sync --extra cuda
```

`uv sync` creates `.venv` and installs the exact versions from `uv.lock`. Prefix commands with
`uv run` to use it, or activate it with `source .venv/bin/activate`.

### Dev container

A [dev container](https://containers.dev/) is provided in [`.devcontainer/`](.devcontainer/). It is
built on `nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04` and declares
`"hostRequirements": { "gpu": "optional" }`, so it uses the GPU when the host exposes one and runs
on the CPU otherwise. "Reopen in Container" from VS Code runs `uv sync --extra cuda --group dev`
and reports whether TensorFlow can see a GPU.

GPU passthrough additionally requires, on the host, a recent NVIDIA driver, the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
and Docker configured with the `nvidia` runtime. On WSL2 install the driver on Windows (not inside
WSL) and enable GPU support in Docker Desktop.

## Usage

### Get started

To get started, you can run the demo code to apply a pre-computed universal perturbation for Inception on the image of your choice:

```bash
uv run python demo_inception.py -i data/test_img.png
```

This will download the pre-trained model, and show the image without and with universal perturbation with the estimated labels.
In this example, the pre-computed targeted universal perturbation in `data/universal.npy` is used. This perturbation targets the kit fox class.

> [!NOTE]
> `data/universal.npy` is not committed to this repository. When it is absent, `demo_inception.py`
> computes a perturbation instead, which needs an ImageNet training set at the path given by `-t`
> (`/datasets2/ILSVRC2012/train` by default).

### Computing a universal perturbation for your model

To compute a universal perturbation for your model, please follow the same structure as in `demo_inception.py`.
In particular, you should use the `universal_perturbation` function (see `universal_pert.py` for details), with the set of training images
used to compute the perturbation, as well as the feedforward and gradient functions.

## Development

```bash
uv sync --group dev        # install the dev tooling
uv run ruff check .        # lint
uv run ruff format .       # format
uv run pytest              # run the test suite
```

Linting, formatting, the test suite (Python 3.11/3.12/3.13), the lockfile, and the package build
all run on every push and pull request against `main`; see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

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

# 日本語ドキュメント

## 概要

このリポジトリは、[Universal Adversarial Perturbation](https://github.com/LTS4/universal)を元とし、より少ない入力情報で、同様の性質を持つ摂動を高速に生産することが可能なアルゴリズムを実装しています。

![prepare_graph](https://user-images.githubusercontent.com/32987034/75492815-e614af00-59fb-11ea-97b7-d61460d2f876.PNG)

上記の表は一般的なUniversal Adversarial Perturbationの生成アルゴリズムとこのリポジトリにおいて用いられているアルゴリズムによって得られる摂動の使用画像数ごとの比較である。
この結果を見ると基本的に提案手法のほうがより少ない枚数でも高いエラー率の摂動を獲得していることが見て取れる。

ただし、完全な上位互換というわけではない。各画像に対する多重度`M`をパラメータとして設定しなくてはならない。多重度`M`は整数値を取り、`M`を変化させた場合のエラー率の推移は次のグラフに示す。実装では多重度`M`は引数`search_dim`に該当する。

![graph](https://user-images.githubusercontent.com/32987034/75492769-c7aeb380-59fb-11ea-86f2-67d1f13eddc3.PNG)

このグラフを見ると、多重度は高ければ高いほど良いというわけではなく画像数に対する適切なパラメータを設定しなくては有効な摂動の生成は難しく、パラメータの探索コストが増えたともいえる。

しかしながら、実験したすべての画像数において、高い既存手法に比べ高いエラー率を出すことのできるパラメータが存在している。
