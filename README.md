# Universal adversarial perturbations for fast and high error rate

This repos extended [Universal Adversarial Perturbation](https://github.com/LTS4/universal) to fast and high error rate in small data.

Only Python3.x ver included.

*python*: Python3.x code to generate universal perturbations using [TensorFlow](https://github.com/tensorflow/tensorflow).Required library is written `requirements.txt` .

## Usage

### Get started

To get started, you can run the demo code to apply a pre-computed universal perturbation for Inception on the image of your choice
```
python demo_inception.py -i data/test_img.png	
```
This will download the pre-trained model, and show the image without and with universal perturbation with the estimated labels.
In this example, the pre-computed targeted universal perturbation in `data/universal.npy` is used. This Perturbation targeted kit fox class.

### Computing a universal perturbation for your model

To compute a universal perturbation for your model, please follow the same struture as in `demo_inception.py`.
In particular, you should use the `universal_perturbation` function (see `universal_pert.py` for details), with the set of training images 
used to compute the perturbation, as well as the feedforward and gradient functions.


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
