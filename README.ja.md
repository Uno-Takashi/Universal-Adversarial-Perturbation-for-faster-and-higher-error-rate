# 少数画像から高速・高エラー率な Universal Adversarial Perturbation

[![CI](https://github.com/Uno-Takashi/Universal-Adversarial-Perturbation-for-faster-and-higher-error-rate/actions/workflows/ci.yml/badge.svg)](https://github.com/Uno-Takashi/Universal-Adversarial-Perturbation-for-faster-and-higher-error-rate/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

*[English README](README.md)*

## 概要

このリポジトリは [Universal Adversarial Perturbation](https://github.com/LTS4/universal) を元とし、
より少ない入力情報で同様の性質を持つ摂動を高速に生成できるアルゴリズムを実装しています。

画像に依存しない単一の摂動 `v` を求め、データセットの多くの画像にそれを加算すると分類器の予測が変わる、
というものです。元のアルゴリズムが各画像を最近傍の決定境界へ動かすのに対し、本手法は各画像について
上位 `M` 個の次点クラスを狙います。この多重度 `M` がコード上の `search_num` 引数で、これにより
より少ない枚数から高いエラー率に到達します。

![prepare_graph](https://user-images.githubusercontent.com/32987034/75492815-e614af00-59fb-11ea-97b7-d61460d2f876.PNG)

上記の表は、一般的な Universal Adversarial Perturbation の生成アルゴリズムと、このリポジトリで
用いられているアルゴリズムによって得られる摂動を、使用画像数ごとに比較したものです。
この結果を見ると、基本的に提案手法のほうがより少ない枚数でも高いエラー率の摂動を獲得していることが
見て取れます。

ただし完全な上位互換というわけではありません。各画像に対する多重度 `M` をパラメータとして設定しなくては
なりません。多重度 `M` は整数値を取り、`M` を変化させた場合のエラー率の推移は次のグラフに示します。

![graph](https://user-images.githubusercontent.com/32987034/75492769-c7aeb380-59fb-11ea-86f2-67d1f13eddc3.PNG)

このグラフを見ると、多重度は高ければ高いほど良いというわけではなく、画像数に対する適切なパラメータを
設定しなくては有効な摂動の生成は難しく、パラメータの探索コストが増えたともいえます。

しかしながら、実験したすべての画像数において、既存手法に比べ高いエラー率を出すことのできるパラメータが
存在しています。

多重度 `M` の掃引は `--search-num` で行えます:

```bash
uv run python experiment.py --models mobilenet_v2 --search-num 1 3 5 10
```

## 必要環境

- Python 3.11 – 3.13
- 依存管理に [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU は任意（無い場合は TensorFlow が CPU で動作します）

## セットアップ

[uv をインストール](https://docs.astral.sh/uv/getting-started/installation/)した上で:

```bash
# CPU、または CUDA と cuDNN がシステムに入っている環境
uv sync

# システムに CUDA が無い GPU 環境（CUDA ライブラリを wheel として取得）
uv sync --extra cuda
```

`uv sync` は `.venv` を作成し、`uv.lock` に固定された正確なバージョンをインストールします。
コマンドは `uv run` を前置するか、`source .venv/bin/activate` で有効化してください。

> [!IMPORTANT]
> `cuda` extra を使う場合は `uv run` にも渡してください（`uv run --extra cuda python ...`）。
> あるいは `UV_NO_SYNC=1` を設定してください。素の `uv run` は既定の extra に再 sync するため
> CUDA wheel をアンインストールします。dev container 内ではベースイメージが CUDA を提供するため
> extra 自体が不要で、この問題は起きません。

### Dev container

[`.devcontainer/`](.devcontainer/) に [dev container](https://containers.dev/) を用意しています。
`nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04` をベースに `"hostRequirements": { "gpu": "optional" }`
を宣言しているため、GPU があるホストでは GPU を使い、無ければ CPU で動作します。VS Code の
「Reopen in Container」で `uv sync --group dev` が走り、TensorFlow が GPU を認識できているかを表示します。

ベースイメージが CUDA 12.6 と cuDNN をシステム全体に提供するので、素の `tensorflow` wheel がそれを
利用します。`cuda` extra はコンテナ**外**の GPU マシン用です。

GPU パススルーには、ホスト側に最近の NVIDIA ドライバ、
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)、
および `nvidia` ランタイムを設定した Docker が必要です。WSL2 では（WSL 内ではなく）Windows 側に
ドライバをインストールし、Docker Desktop で GPU サポートを有効にしてください。

## 使い方

### デモ: 1枚の画像に摂動を適用する

```bash
uv run python demo_inception.py -i data/test_img.png
```

Inception 5h モデルをダウンロードし、Universal Adversarial Perturbation を加える前後の画像を
推定クラス名付きで表示します。`-o figure.png` を付けるとウィンドウを開かずファイルに保存します。

> [!NOTE]
> `data/universal.npy` はこのリポジトリにコミットされていません。存在しない場合、
> `demo_inception.py` は先に摂動を計算します。その際 Hugging Face から画像をストリーミングします
> （枚数は `-n` で指定）。ローカルの ImageNet ディレクトリを使う場合は
> `-t /path/to/ILSVRC2012/train` を渡してください。

### 複数モデルでの実験

[`experiment.py`](experiment.py) はアルゴリズムを一通り実行し、fooling rate を報告します。
必要なもの（モデルの重み、ILSVRC 画像）はすべて自動でダウンロードされるため、`uv sync` だけの
新規マシンでも実験を再現できます。

```bash
# 単一モデル
uv run python experiment.py --models inception5h --num-images 100

# Inception 以外でも有効か確認する
uv run python experiment.py --models inception5h mobilenet_v2 resnet50 --num-images 100 --transfer

# 多重度 M の掃引
uv run python experiment.py --models mobilenet_v2 --search-num 1 3 5 10

# Hugging Face ではなくローカルの ILSVRC を使う
uv run python experiment.py --dataset /datasets2/ILSVRC2012/train --num-images 500
```

実行ごとに表を出力し、`--out DIR` を付けると各摂動を `.npy` として `summary.json` とともに保存します:

```
model               M  images   clean    fool  fool(clip)    sec
----------------------------------------------------------------
inception5h         3       8   75.0%   87.5%       87.5%      7
mobilenet_v2        3       8   72.0%  100.0%      100.0%      7
resnet50            3       8   87.5%  100.0%      100.0%     18
```

`fool` は予測ラベルが変化した画像の割合で、アルゴリズムが最適化する基準そのものです。
`fool(clip)` は摂動後の画像を `[0, 255]` にクリップしたうえでの同じ指標、つまり画像として妥当な
範囲に収めた場合の値です。`--transfer` を付けると、各摂動を入力サイズが一致する他モデルに対しても
評価します。

### 対応モデル

| 名前 | 取得元 | 入力 | クラス数 |
|---|---|---|---|
| `inception5h` | frozen graph（論文のモデル） | 224×224 | 1008 |
| `inception_v3` | `tf.keras.applications` | 299×299 | 1000 |
| `resnet50` | `tf.keras.applications` | 224×224 | 1000 |
| `mobilenet_v2` | `tf.keras.applications` | 224×224 | 1000 |
| `vgg16` | `tf.keras.applications` | 224×224 | 1000 |
| `efficientnet_b0` | `tf.keras.applications` | 224×224 | 1000 |
| `convnext_tiny` | `tf.keras.applications` | 224×224 | 1000 |

独自モデルを追加する場合は [`classifiers.Classifier`](classifiers.py) を継承し、`_forward`
（生ピクセルから softmax 前のロジットへ）と `label` を実装してください。勾配は `tf.GradientTape`
により自動的に得られます。

### データセット

画像は Hugging Face の dataset viewer 経由で取得するため、要求した枚数だけがダウンロードされます。
50枚の実験なら数百 MB のシャードではなく50枚分だけです。

| 短縮名 | データセット | 備考 |
|---|---|---|
| `imagenet-val`（既定） | `NightMachinery/ImageNet1K-val-indexed` | ILSVRC2012 validation、認証不要 |
| `imagenet-val-alt` | `danaroth/imagenet_val` | 認証不要のミラー |
| `imagenet-val-224` | `danjacobellis/imagenet_1k_val_224` | 224 に事前クロップ済み |
| `imagenet-1k` | `imagenet-1k` | gated: `HF_TOKEN` 設定と規約同意が必要 |

短縮名の代わりに任意の hub id が使えます。ローカルの ILSVRC 形式ディレクトリ（クラスごとに
サブディレクトリ）のパスも指定できます。

### 自分のコードで摂動を計算する

```python
from classifiers import build_classifier
from imagenet_source import load_images
from universal_pert import universal_perturbation

model = build_classifier("resnet50")
dataset, _ = load_images("imagenet-val", num_images=50, image_size=model.image_size)

v = universal_perturbation(dataset, model, model.gradients, delta=0.2, search_num=5)
```

`universal_perturbation` は順伝播の callable と勾配の callable を受け取るため、微分可能な
任意のモデルで動作します。パラメータは [`universal_pert.py`](universal_pert.py) を参照してください。

## 設計上の注意

**生ピクセル空間。** すべての分類器は `[0, 255]` の生 RGB を受け取り、前処理は TensorFlow グラフ
内部で行います。これにより摂動と `xi` の予算がピクセル値の単位で表現され、モデルをまたいで同じ数値が
同じ意味を持ちます。またクリッピングが単純な `clip(raw + v, 0, 255)` になります。Inception 5h に
ついては、元の平均引き定式化と等価です。定数の減算は平行移動なので勾配と l_p ノルムを変えず、
`data/universal.npy` もそのまま有効です。

**TF2 ネイティブ。** モデルは eager な callable で、勾配は `tf.GradientTape` から得ます。
session・placeholder・feed dict は存在せず、`tf.while_loop` による手書きヤコビアンもありません。
[`classifiers.py`](classifiers.py) の `tf.compat.v1.wrap_function` が唯一残る compat 呼び出しで
（frozen `GraphDef` には他の TF2 入口がないため）、
[`tests/test_tf2_native.py`](tests/test_tf2_native.py) がそれが唯一であることを検査しています。

**モジュール構成。**

| モジュール | 役割 |
|---|---|
| [`deepfool.py`](deepfool.py) | 画像ごとの非標的型攻撃 |
| [`deeptarget.py`](deeptarget.py) | 画像ごとの標的型攻撃。本手法の内側ループ |
| [`universal_pert.py`](universal_pert.py) | Universal Adversarial Perturbation のアルゴリズム |
| [`classifiers.py`](classifiers.py) | モデル抽象と実装 |
| [`imagenet_source.py`](imagenet_source.py) | Hugging Face またはディスクからの画像読み込み |
| [`experiment.py`](experiment.py) | 実験ドライバと結果報告 |
| [`util_univ.py`](util_univ.py) | fooling rate の指標とクリッピング補助 |
| [`prepare_imagenet_data.py`](prepare_imagenet_data.py) | 旧来の平均引き前処理の補助関数 |

アルゴリズム部分は純粋な NumPy で callable を受け取るため、TensorFlow に依存しません。

## 開発

```bash
uv sync --group dev        # 開発ツールをインストール
uv run ruff check .        # lint
uv run ruff format .       # フォーマット
uv run pytest              # テスト実行
```

lint、フォーマット、テスト（Python 3.11/3.12/3.13）、lockfile、パッケージビルドが `main` への
push と pull request ごとに実行されます。[`.github/workflows/ci.yml`](.github/workflows/ci.yml) を
参照してください。テストはネットワークもモデルの重みも必要としません。アルゴリズムは解析的な
ヤコビアンを持つ合成線形分類器で検証し、Hugging Face のローダは疑似 HTTP 層で検証しています。

## リリース

リリースはタグから作成します。`pyproject.toml` の `version` を上げてから:

```bash
git tag v0.2.0
git push origin v0.2.0
```

[`.github/workflows/release.yml`](.github/workflows/release.yml) がタグとプロジェクトバージョンの
一致を検証し、テストを実行し、sdist と wheel をビルドして、それらを添付した GitHub Release を
公開します。PyPI への公開はオプトインです。リポジトリ変数 `PUBLISH_TO_PYPI` を `true` にし、
`pypi` environment に対して PyPI trusted publishing を設定してください。

## 参考文献

[1] S. Moosavi-Dezfooli\*, A. Fawzi\*, O. Fawzi, P. Frossard:
[*Universal adversarial perturbations*](http://arxiv.org/pdf/1610.08401), CVPR 2017.
