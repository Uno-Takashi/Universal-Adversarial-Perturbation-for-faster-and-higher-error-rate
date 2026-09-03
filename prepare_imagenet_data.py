import os

import numpy as np
from PIL import Image

CLASS_INDEX = None
CLASS_INDEX_PATH = (
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
)

# Empirical channel means of the ILSVRC2012 training set (RGB order).
CHANNEL_MEANS = (123.68, 116.779, 103.939)


def _load_rgb(im_path, img_size=None):
    """Load an image as an RGB array, optionally resized to ``img_size`` (rows, cols)."""
    with Image.open(im_path) as img:
        img = img.convert("RGB")
        if img_size:
            # PIL takes (width, height); img_size follows the (rows, cols) convention.
            img = img.resize((img_size[1], img_size[0]), Image.BILINEAR)
        return np.asarray(img)


def preprocess_image_batch(image_paths, img_size=None, crop_size=None, color_mode="rgb", out=None):
    img_list = []

    for im_path in image_paths:
        img = _load_rgb(im_path, img_size).astype("float32")
        # We normalize the colors (in RGB space) with the empirical means on the training set
        img = do_image_avg(img)
        # We permute the colors to get them in the BGR order
        # if color_mode=="bgr":
        #    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]

        if crop_size:
            img = img[
                (img_size[0] - crop_size[0]) // 2 : (img_size[0] + crop_size[0]) // 2,
                (img_size[1] - crop_size[1]) // 2 : (img_size[1] + crop_size[1]) // 2,
                :,
            ]

        img_list.append(img)

    try:
        img_batch = np.stack(img_list, axis=0)
    except ValueError as exc:
        raise ValueError(
            "when img_size and crop_size are None, images in image_paths must have the same shapes."
        ) from exc

    if out is not None and hasattr(out, "append"):
        out.append(img_batch)
        return None
    return img_batch


def undo_image_avg(img):
    img_copy = np.copy(img)
    for channel, mean in enumerate(CHANNEL_MEANS):
        img_copy[:, :, channel] = img_copy[:, :, channel] + mean
    return img_copy


def do_image_avg(img):
    img_copy = np.copy(img)
    for channel, mean in enumerate(CHANNEL_MEANS):
        img_copy[:, :, channel] = img_copy[:, :, channel] - mean
    return img_copy


def undo_image_list(img_list):
    """Add the channel means back and clip into uint8.

    Clipping matters: a perturbed image easily leaves [0, 255], and an unclipped cast wraps
    modulo 256, turning an overshoot of 300 into 44.
    """
    undo_list = np.zeros(img_list.shape, dtype=np.uint8)
    for x in range(undo_list.shape[0]):
        undo_list[x] = np.clip(undo_image_avg(img_list[x]), 0, 255).astype(np.uint8)
    return undo_list


def do_image_list(img_list):
    do_list = np.zeros(img_list.shape, dtype=np.float32)
    for x in range(do_list.shape[0]):
        do_list[x] = do_image_avg(img_list[x]).astype(np.float32)
    return do_list


def create_imagenet_npy(path_train_imagenet, len_batch=10000):
    # path_train_imagenet = '/datasets2/ILSVRC2012/train'

    sz_img = [224, 224]
    num_channels = 3
    num_classes = 1000

    im_array = np.zeros([len_batch, *sz_img, num_channels], dtype=np.float32)
    num_imgs_per_batch = int(len_batch / num_classes)

    dirs = [x[0] for x in os.walk(path_train_imagenet)]
    dirs = dirs[1:]

    # Sort the directory in alphabetical order (same as synset_words.txt)
    dirs = sorted(dirs)

    it = 0
    files_per_class = [0 for x in range(num_classes)]

    for d in dirs:
        for _, _, filename in os.walk(os.path.join(path_train_imagenet, d)):
            files_per_class[it] = filename
        it = it + 1

    it = 0
    # Load images, pre-process, and save
    for k in range(num_classes):
        for u in range(num_imgs_per_batch):
            print("Processing image number ", it)
            path_img = os.path.join(dirs[k], files_per_class[k][u])
            image = preprocess_image_batch(
                [path_img], img_size=(256, 256), crop_size=(224, 224), color_mode="rgb"
            )
            im_array[it : (it + 1), :, :, :] = image
            it = it + 1

    return im_array
