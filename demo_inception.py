"""Apply a universal adversarial perturbation to one image, using Inception 5h.

This is the original demo, ported to TF2: the model is loaded once as an eager callable (see
:mod:`classifiers`) and there is no session, placeholder or feed dict anywhere. Images are
raw ``[0, 255]`` RGB throughout, which leaves ``data/universal.npy`` valid -- the mean
subtraction it used to be expressed against is a shift, so it does not change a perturbation.
"""

import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np

from classifiers import Inception5hClassifier
from imagenet_source import load_images, resize_and_crop
from universal_pert import universal_perturbation
from util_univ import clip_perturbed

NUM_CLASSES = 2


def load_image(path, image_size):
    from PIL import Image

    with Image.open(path) as handle:
        return np.asarray(resize_and_crop(handle, image_size), dtype=np.float32)[None]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply a universal adversarial perturbation to an image using Inception."
    )
    parser.add_argument(
        "-i",
        "--test_image",
        default=os.path.join("data", "test_img.png"),
        help="path to the image the perturbation is demonstrated on",
    )
    parser.add_argument(
        "-t",
        "--training_path",
        default=None,
        help="ImageNet training directory, or a Hugging Face dataset id, used only when a "
        "perturbation has to be computed (default: stream from Hugging Face)",
    )
    parser.add_argument(
        "-n",
        "--num_images",
        type=int,
        default=50,
        help="how many images to compute the perturbation from",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="save the figure here instead of opening a window",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = Inception5hClassifier()
    file_perturbation = os.path.join("data", "universal.npy")

    if not os.path.isfile(file_perturbation):
        print(">> No pre-computed perturbation found; computing one.")
        print(f">> Loading {args.num_images} images...")
        dataset, _ = load_images(
            args.training_path or "imagenet-val",
            num_images=args.num_images,
            image_size=model.image_size,
        )
        v = universal_perturbation(
            dataset, model, model.gradients, delta=0.1, num_classes=NUM_CLASSES
        )
        np.save(file_perturbation, v)
    else:
        print(
            ">> Found a pre-computed universal perturbation! "
            f"Retrieving it from {file_perturbation}"
        )
        v = np.load(file_perturbation)

    print(">> Testing the universal perturbation on an image")
    image_original = load_image(args.test_image, model.image_size)
    image_perturbed = clip_perturbed(image_original, v)

    str_label_original = model.label(model.predict(image_original))
    str_label_perturbed = model.label(model.predict(image_perturbed))
    print(f">> {str_label_original} --> {str_label_perturbed}")

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_original[0].astype(np.uint8))
    plt.title(str_label_original)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image_perturbed[0].astype(np.uint8))
    plt.title(str_label_perturbed)
    plt.axis("off")

    if args.output:
        plt.savefig(args.output, bbox_inches="tight", dpi=150)
        print(f">> Figure written to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
