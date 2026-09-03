import os

import matplotlib.pyplot as plt
import numpy as np

from prepare_imagenet_data import undo_image_avg, undo_image_list


def visualization_pert(v):
    plt.imshow(v)
    plt.show()


def img2str(f, img):
    num_pert = np.argmax(f(img), axis=1).flatten()
    return cat2label_str(num_pert)


def cat2label_str(num_pert):
    """Map a class index to its human-readable label.

    Accepts a scalar or a single-element array; numpy >= 2 refuses to convert the latter
    with a plain ``int()``.
    """
    index = int(np.ravel(num_pert)[0])
    with open(os.path.join("data", "labels.txt")) as fh:
        labels = fh.read().split("\n")
    return labels[index - 1].split(",")[0]


def avg_add_clip_pert(avg_img, v):
    clipped_v = np.clip(undo_image_avg(avg_img[0, :, :, :] + v[0, :, :, :]), 0, 255) - np.clip(
        undo_image_avg(avg_img[0, :, :, :]), 0, 255
    )
    pert_img = avg_img + clipped_v[None, :, :, :]
    return pert_img


def _batched_labels(f, dataset, batch_size, undo_avg=False):
    """Return the estimated label of every image in ``dataset``, computed batch by batch."""
    num_images = np.shape(dataset)[0]
    est_labels = np.zeros(num_images)
    num_batches = int(np.ceil(float(num_images) / float(batch_size)))

    for ii in range(num_batches):
        m = ii * batch_size
        upper = min((ii + 1) * batch_size, num_images)
        batch = dataset[m:upper, :, :, :]
        if undo_avg:
            batch = undo_image_list(batch)
        est_labels[m:upper] = np.argmax(f(batch), axis=1).flatten()

    return est_labels


def fooling_rate_calc(v, dataset, f, batch_size=100):
    dataset_perturbed = dataset + v
    num_images = np.shape(dataset)[0]

    est_labels_orig = _batched_labels(f, dataset, batch_size, undo_avg=True)
    est_labels_pert = _batched_labels(f, dataset_perturbed, batch_size, undo_avg=True)

    return float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))


def target_fooling_rate_calc(v, dataset, f, target, batch_size=100):
    dataset_perturbed = dataset + v
    num_images = np.shape(dataset)[0]

    est_labels_pert = _batched_labels(f, dataset_perturbed, batch_size)

    return float(np.sum(est_labels_pert == target) / float(num_images))


def fooling_rate_calc_all(v, dataset, f, target, batch_size=100):
    dataset_perturbed = dataset + v
    num_images = np.shape(dataset)[0]

    est_labels_orig = _batched_labels(f, dataset, batch_size)
    est_labels_pert = _batched_labels(f, dataset_perturbed, batch_size)

    fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))
    target_fooling_rate = float(np.sum(est_labels_pert == target) / float(num_images))
    return fooling_rate, target_fooling_rate
