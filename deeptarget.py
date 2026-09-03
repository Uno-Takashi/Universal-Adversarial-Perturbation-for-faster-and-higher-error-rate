import numpy as np


def deeptarget(image, f, grads, overshoot=0.02, max_iter=50, target=None):
    """Find a perturbation that pushes the classifier towards ``target``.

    :param image: Image of size HxWx3
    :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
    :param grads: gradient functions with respect to input (as many gradients as classes).
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02)
    :param max_iter: maximum number of iterations for deeptarget (default = 50)
    :param target: index of the class the perturbation should aim for (required)
    :return: minimal perturbation that fools the classifier, number of iterations that it
        required, new estimated_label and perturbed image
    """
    if target is None:
        raise ValueError("Target is not enough")

    f_image = np.array(f(image)).flatten()
    order = f_image.argsort()[::-1]

    order = order[0:2]
    label = order[0]
    order[1] = target

    input_shape = image.shape
    pert_image = image

    f_i = np.array(f(pert_image)).flatten()
    k_i = int(np.argmax(f_i))

    r_tot = np.zeros(input_shape)

    loop_i = 0

    # We can't use k_i != target. DeepTarget is limited to the L2 norm and would
    # otherwise generate a huge perturbation.
    while k_i == label and loop_i < max_iter:
        gradients = np.asarray(grads(pert_image, order))

        # set new w and new f_t
        w = gradients[1, :, :, :, :] - gradients[0, :, :, :, :]
        f_t = f_i[order[1]] - f_i[order[0]]
        pert = abs(f_t) / np.linalg.norm(w.flatten())

        # compute r_i and r_tot
        r_i = pert * w / np.linalg.norm(w)
        r_tot = r_tot + r_i

        # compute new perturbed image
        pert_image = image + (1 + overshoot) * r_tot
        loop_i += 1

        # compute new label
        f_i = np.array(f(pert_image)).flatten()
        k_i = int(np.argmax(f_i))

    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, k_i, pert_image
