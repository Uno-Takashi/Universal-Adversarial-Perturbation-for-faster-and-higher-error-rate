import argparse
import os.path
import zipfile
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from prepare_imagenet_data import create_imagenet_npy, preprocess_image_batch, undo_image_avg
from universal_pert import universal_perturbation
from util_univ import avg_add_clip_pert, cat2label_str

# The Inception 5h graph is a TensorFlow 1.x frozen GraphDef, so it is driven through
# the compat.v1 session API rather than eager execution.
tf.compat.v1.disable_eager_execution()

DEVICE = "/gpu:0"
NUM_CLASSES = 2
INCEPTION_URL = "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"


def jacobian(y_flat, x, inds):
    n = NUM_CLASSES  # Not really necessary, just a quick fix.
    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=n),
    ]
    _, jacobian_stack = tf.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j + 1, result.write(j, tf.gradients(y_flat[inds[j]], x))),
        loop_vars,
    )
    return jacobian_stack.stack()


def download_inception_model(inception_model_path):
    print("Downloading Inception model...")
    archive = os.path.join("data", "inception5h.zip")
    urlretrieve(INCEPTION_URL, archive)
    with zipfile.ZipFile(archive, "r") as zip_ref:
        zip_ref.extract("tensorflow_inception_graph.pb", "data")
    return inception_model_path


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
        default="/datasets2/ILSVRC2012/train",
        help="path to the ImageNet training set, used only when a perturbation must be computed",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    path_train_imagenet = args.training_path
    path_test_image = args.test_image

    with tf.device(DEVICE):
        persisted_sess = tf.compat.v1.Session()
        inception_model_path = os.path.join("data", "tensorflow_inception_graph.pb")

        if not os.path.isfile(inception_model_path):
            download_inception_model(inception_model_path)

        # Load the Inception model
        with tf.io.gfile.GFile(inception_model_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            persisted_sess.graph.as_default()
            tf.import_graph_def(graph_def, name="")

        persisted_input = persisted_sess.graph.get_tensor_by_name("input:0")
        persisted_output = persisted_sess.graph.get_tensor_by_name("softmax2_pre_activation:0")

        print(">> Computing feedforward function...")

        def f(image_inp):
            return persisted_sess.run(
                persisted_output,
                feed_dict={persisted_input: np.reshape(image_inp, (-1, 224, 224, 3))},
            )

        file_perturbation = os.path.join("data", "universal.npy")

        if not os.path.isfile(file_perturbation):
            # TODO: Optimize this construction part!
            print(">> Compiling the gradient tensorflow functions. This might take some time...")
            y_flat = tf.reshape(persisted_output, (-1,))
            inds = tf.compat.v1.placeholder(tf.int32, shape=(NUM_CLASSES,))
            dydx = jacobian(y_flat, persisted_input, inds)

            print(">> Computing gradient function...")

            def grad_fs(image_inp, indices):
                return persisted_sess.run(
                    dydx, feed_dict={persisted_input: image_inp, inds: indices}
                ).squeeze(axis=1)

            # Load/Create data
            datafile = os.path.join("data", "imagenet_data.npy")
            if not os.path.isfile(datafile):
                print(">> Creating pre-processed imagenet data...")
                X = create_imagenet_npy(path_train_imagenet)

                # Caution: saving this can take a lot of space, so it is left commented out.
                # np.save(datafile, X)
            else:
                print(">> Pre-processed imagenet data detected")
                X = np.load(datafile)

            # Running universal perturbation
            v = universal_perturbation(X, f, grad_fs, delta=0.1, num_classes=NUM_CLASSES)

            # Saving the universal perturbation
            np.save(file_perturbation, v)
        else:
            print(
                ">> Found a pre-computed universal perturbation! "
                f"Retrieving it from {file_perturbation}"
            )
            v = np.load(file_perturbation)

        print(">> Testing the universal perturbation on an image")

        image_original = preprocess_image_batch(
            [path_test_image], img_size=(256, 256), crop_size=(224, 224), color_mode="rgb"
        )
        label_original = np.argmax(f(image_original), axis=1).flatten()
        str_label_original = cat2label_str(label_original)

        # Clip the perturbation to make sure images fit in uint8
        image_perturbed = avg_add_clip_pert(image_original, v)
        label_perturbed = np.argmax(f(image_perturbed), axis=1).flatten()
        str_label_perturbed = cat2label_str(label_perturbed)

        # Show original and perturbed image
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(undo_image_avg(image_original[0, :, :, :]).astype(dtype="uint8"))
        plt.title(str_label_original)

        plt.subplot(1, 2, 2)
        plt.imshow(undo_image_avg(image_perturbed[0, :, :, :]).astype(dtype="uint8"))
        plt.title(str_label_perturbed)

        plt.show()


if __name__ == "__main__":
    main()
