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

|    画像枚数    |    100    |    500    |    1000    |    4000    |
|:--------------:|:---------:|:---------:|:----------:|:----------:|
|     元論文     |    10%    |    27%    |     41%    |     68%    |
|    提案手法    |    39%    |    68%    |     74%    |     NaN    |

上記の表を見るとこのリポジトリで提案している手法では500枚の画像を生成に用いた場合のエラー率が68%に対して、元論文の手法による実装では同様のエラー率を得るのに4000枚の画像を必要とします。
このリポジトリのアルゴリズムではより効率的なUniversal Adversarial Perturbationの生成を可能にします。
