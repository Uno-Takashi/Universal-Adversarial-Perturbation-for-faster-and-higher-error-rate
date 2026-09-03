import numpy as np

from deeptarget import deeptarget
from util_univ import fooling_rate_calc

_rng = np.random.default_rng()


def proj_lp(v, xi, p):
    """Project ``v`` on the l_p ball centered at 0 and of radius ``xi``.

    Only p = 2 and p = Inf are supported.
    """
    if p == 2:
        v = v * min(1, xi / np.linalg.norm(v.flatten()))
        # v = v / np.linalg.norm(v.flatten()) * xi
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
        raise ValueError("Values of p different from 2 and Inf are currently not supported...")

    return v


def universal_perturbation(
    dataset,
    f,
    grads,
    delta=0.2,
    max_iter_uni=10,
    xi=10,
    p=np.inf,
    num_classes=10,
    overshoot=0.02,
    max_iter_df=25,
    search_num=5,
    batch_size=128,
):
    """Compute a universal adversarial perturbation for the classifier ``f``.

    :param dataset: Images of size MxHxWxC (M: number of images)
    :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
    :param grads: gradient functions with respect to input (as many gradients as classes).
    :param delta: controls the desired fooling rate (default = 80% fooling rate)
    :param max_iter_uni: optional other termination criterion (maximum number of passes over
        the dataset, default = 10)
    :param xi: controls the l_p magnitude of the perturbation (default = 10)
    :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!)
        (default = np.inf)
    :param num_classes: limits the number of classes to test against (default = 10)
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02)
    :param max_iter_df: maximum number of iterations for deeptarget (default = 25)
    :param search_num: multiplicity M, i.e. how many runner-up classes are targeted per image
    :param batch_size: batch size used when evaluating the fooling rate
    :return: the universal perturbation.
    """
    v = 0
    fooling_rate = 0.0
    num_images = np.shape(dataset)[0]  # The images should be stacked ALONG FIRST DIMENSION

    itr = 0
    while fooling_rate < 1 - delta and itr < max_iter_uni:
        # Shuffle the dataset
        _rng.shuffle(dataset)

        print("Starting pass number ", itr)

        # Go through the data set and compute the perturbation increments sequentially
        for k in range(num_images):
            cur_img = dataset[k : (k + 1), :, :, :]

            if int(np.argmax(np.array(f(cur_img)).flatten())) == int(
                np.argmax(np.array(f(cur_img + v)).flatten())
            ):
                img_temp = cur_img + v
                order = np.array(f(cur_img + v)).flatten().argsort()[::-1]

                order = order[1 : search_num + 1]
                print(">> k = ", k, ", pass #", itr)
                for x in order:
                    print(str(np.argmax(np.array(f(img_temp)).flatten())) + " ---> " + str(x))

                    # Compute adversarial perturbation
                    dr, n_iter, _ki, _ = deeptarget(
                        img_temp,
                        f,
                        grads,
                        overshoot=overshoot,
                        max_iter=max_iter_df,
                        target=int(x),
                    )
                    # Make sure it converged...
                    if n_iter < max_iter_df - 1:
                        v = v + dr

                        # Project on l_p ball
                        v = proj_lp(v, xi, p)

        itr = itr + 1

        # Compute the fooling rate
        fooling_rate = fooling_rate_calc(v=v, dataset=dataset, f=f, batch_size=batch_size)
        print("")
        print("FOOLING RATE = ", fooling_rate)

    return v
