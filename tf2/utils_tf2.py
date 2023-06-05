import os

import numpy as np
import eagerpy as ep
import tensorflow as tf
from typing import Tuple, List
from PIL import Image


def gkern(kernlen=21, nsig=3):
    """
    Returns a 2D Gaussian kernel array.
    这是一个副本，来自于原CA2 TF1版本中utils.py
    """

    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)

    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3)
    return stack_kernel


def optimize_linear(grad, eps, norm=np.inf):
    """
    这是来自cleverhans项目的一个副本

    项目地址：https://github.com/cleverhans-lab/cleverhans/blob/master/cleverhans/tf2/utils.py

    Solves for the optimal input to a linear function under a norm constraint.

    Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)

    :param grad: tf tensor containing a batch of gradients
    :param eps: float scalar specifying size of constraint region
    :param norm: int specifying order of norm
    :returns:
      tf tensor containing optimal perturbation
    """

    # Convert the iterator returned by `range` into a list.
    axis = list(range(1, len(grad.get_shape())))
    avoid_zero_div = 1e-12
    if norm == np.inf:
        # Take sign of gradient
        optimal_perturbation = tf.sign(grad)
        # The following line should not change the numerical results. It applies only because
        # `optimal_perturbation` is the output of a `sign` op, which has zero derivative anyway.
        # It should not be applied for the other norms, where the perturbation has a non-zero derivative.
        optimal_perturbation = tf.stop_gradient(optimal_perturbation)
    elif norm == 1:
        abs_grad = tf.abs(grad)
        sign = tf.sign(grad)
        max_abs_grad = tf.reduce_max(abs_grad, axis, keepdims=True)
        tied_for_max = tf.dtypes.cast(
            tf.equal(abs_grad, max_abs_grad), dtype=tf.float32
        )
        # tf.reduce_sum: https://www.w3cschool.cn/tensorflow_python/tensorflow_python-5y4d2i2n.html
        num_ties = tf.reduce_sum(tied_for_max, axis, keepdims=True)
        optimal_perturbation = sign * tied_for_max / num_ties
    elif norm == 2:
        square = tf.maximum(
            avoid_zero_div, tf.reduce_sum(tf.square(grad), axis, keepdims=True)
        )
        optimal_perturbation = grad / tf.sqrt(square)
    else:
        raise NotImplementedError(
            "Only L-inf, L1 and L2 norms are currently implemented."
        )

    # Scale perturbation to be the solution for the norm=eps rather than norm=1 problem
    scaled_perturbation = tf.multiply(eps, optimal_perturbation)
    return scaled_perturbation


def clip_eta(eta, norm, eps):
    """
    这是来自cleverhans项目的一个副本

    项目地址：https://github.com/cleverhans-lab/cleverhans/blob/master/cleverhans/tf2/utils.py

    Helper function to clip the perturbation to epsilon norm ball.
    :param eta: A tensor with the current perturbation.
    :param norm: Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param eps: Epsilon, bound of the perturbation.
    """

    # Clipping perturbation eta to self.norm norm ball
    if norm not in [np.inf, 1, 2]:
        raise ValueError("norm must be np.inf, 1, or 2.")
    axis = list(range(1, len(eta.get_shape())))
    avoid_zero_div = 1e-12
    if norm == np.inf:
        eta = tf.clip_by_value(eta, -eps, eps)
    else:
        if norm == 1:
            raise NotImplementedError("")
            # This is not the correct way to project on the L1 norm ball:
            # norm = tf.maximum(avoid_zero_div, reduce_sum(tf.abs(eta), reduc_ind, keepdims=True))
        elif norm == 2:
            # avoid_zero_div must go inside sqrt to avoid a divide by zero in the gradient through this operation
            norm = tf.sqrt(
                tf.maximum(
                    avoid_zero_div, tf.reduce_sum(tf.square(eta), axis, keepdims=True)
                )
            )
        # We must *clip* to within the norm ball, not *normalize* onto the surface of the ball
        factor = tf.minimum(1.0, tf.math.divide(eps, norm))
        eta = eta * factor
    return eta


def samples(
        index: int,
        batchsize: int,
        shape: Tuple[int, int] = None,
        data_format: str = "channels_last",
        bounds: Tuple[float, float] = (0, 255),
        paths: List = None
):
    """
    基于foolbox.utils._samples.
    :param index: 待读取的图像起始索引
    :param batchsize: 待读取图像的数量
    :param shape: 如需调整图像大小，设置shape为目标大小，如(224,224)
    :param data_format: 默认为channels_last
    :param bounds: 数值范围，默认为(0, 255)
    :param paths: 待读取的图像文件路径列表
    :returns:
      图像张量序列
    """

    images = []

    if index + batchsize > len(paths):
        raise ValueError("index + batchsize must <= len(path)")

    for idx in range(index, index + batchsize):  # get filename and label
        file = paths[idx]

        # open file
        image = Image.open(file)

        if shape is not None:
            image = image.resize(shape)

        image = np.asarray(image, dtype=np.float32)

        if image.ndim == 2:
            image = image[..., np.newaxis]

        assert image.ndim == 3

        if data_format == "channels_first":
            image = np.transpose(image, (2, 0, 1))

        images.append(image)

    images_ = np.stack(images)

    if bounds != (0, 255):
        images_ = images_ / 255 * (bounds[1] - bounds[0]) + bounds[0]
    return images_


def is_adv(original_label, adv_label):
    if np.argmax(ep.astensor(original_label).numpy()) != np.argmax(ep.astensor(adv_label).numpy()):
        return True
    return False


def save_image(image, filename, shape=None):
    # from tensor tpye to file
    image = ep.astensor(image).numpy()
    image = Image.fromarray(np.uint8(image))
    if shape is not None:
        image.resize(shape)
    image.save(filename)


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    stack_kernel = gkern(1, 3)
    print(stack_kernel.shape)
    print(stack_kernel)

    print("-- 初始化模型: ResNet50 --")
    model = tf.keras.applications.ResNet50(weights="imagenet")
    bounds = (0, 255)

    print("-- 使用samples读取攻击目标图像 --")
    paths = ["../ref/imagenet_06_609.jpg", "../ref/imagenet_01_559.jpg"]
    images = samples(bounds=bounds, batchsize=1, index=1, paths=paths, shape=(224, 224))

    original_label = model(images)
    print("使用LocalModel预测的目标图像的top标签: ", np.argmax(original_label))
    print('使用LocalModel预测的目标图像的top分类:',
          tf.keras.applications.resnet50.decode_predictions(model.predict(images))[0])
