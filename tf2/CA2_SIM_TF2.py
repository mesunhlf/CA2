"""
Tensorflow 2.0版本的CA2-SIM+版实现，与原TF1版的的CA2-SIM.py对应

经过测试的TensorFlow版本：2.7.0 、 2.8.0

@Skydiver
"""

# 导入全局配置，最大噪声范围、RO/SA策略、TIM、SIM、DIM配置等
from conf_plus import *
from utils_tf2 import *

import math


def ca2_tf2(
        model_fn,
        x,
        norm=np.inf,
        clip_min=None,
        clip_max=None,
        y=None,
        targeted=False,
        sanity_checks=True,
):
    """
    Tensorflow 2.0版本的CA2-SIM+版实现，与原TF1版的的CA2-SIM.py对应
    该实现中基于动量的优化部分参考cleverhans中的momentum_iterative_method. 所有调整的部分，均基于尊重MIM和CA2原文实现
        https://github.com/cleverhans-lab/cleverhans/blob/master/cleverhans/tf2/attacks/momentum_iterative_method.py
    :param model_fn: 本地模型函数，调用可返回输入样本的logits；当前仅支持攻击单个模型，暂不支持攻击集成的多个模型
    :param x: 输入样本
    :param norm: (可选) 范数设置，当前仅支持无穷范数 tf.inf
    :param clip_min: (可选 float) 有效样本下界，用于保证样本不超出预定范围
    :param clip_max: (可选 float) 有效样本上届，用于保证样本不超出预定范围
    :param y: (可选) 当进行targeted攻击时，y设置为targeted标签（格式要求同：tf.argmax(model(images_target), 1)）
    :param targeted: (可选 bool) 当进行targeted攻击时，设置为True
    :param sanity_checks: bool, 设置为True时则进行样本越界检查
    :return: 攻击后获得的对抗样本
    """

    if norm not in [np.inf]:
        raise ValueError("norm当前仅支持 np.inf.")

    asserts = []

    # 如果已设置样本上下界，则进行检查
    if clip_min is not None:
        asserts.append(tf.math.greater_equal(x, clip_min))

    if clip_max is not None:
        asserts.append(tf.math.less_equal(x, clip_max))

    if sanity_checks:
        assert np.all(asserts)

    # 此处处理target/non-target攻击区别，如果是target攻击，y由外部传入；如果是non-target攻击，y由本地模型预测得到
    # y是后续计算损失函数及梯度的必须参数
    if y is None:
        y = tf.argmax(model_fn(x), 1)

    # DIM策略为串联的数据增强类的策略，新增transformation_dim()，实现该增强策略
    # 观察TF1版本CA2-SIM.py中DIM相关实现，DIM策略实施在最接近logits计算的位置，即所有样本在计算logits前都要进行DIM策略变换
    # 所以新增如下结合DIM策略的新model函数，在调用compute_gradient_ca2()时作为参数传入
    def model_fn_dim(x_dim):
        return model_fn(transformation_dim(x_dim))

    # 初始化循环优化动量/梯度
    momentum = tf.zeros_like(x)
    adv_x = tf.zeros_like(x)

    for i in range(phase_num):
        # 进入第i+1个循环优化阶段
        print("---- 进入第[%d]个循环优化阶段, 当前阶段迭代总数为[%d] ----" % (i + 1, phase_step[i]))

        # 每个循环优化阶段开始时，样本均初始化为原始样本（即重新出发）
        adv_x = x
        # 根据循环优化策略，动量初始化为上一阶段循环优化动量 * 学习权重
        momentum = momentum * momentum_learn_factor

        for j in range(phase_step[i]):
            # 进入第i+1个循环优化阶段的第j+1次迭代
            print("--- 进入第[%d]个循环优化阶段的第[%d]次迭代 ---" % (i + 1, j + 1))

            # 计算SA(D2A、VME、DIM、SIM、TIM)策略综合梯度
            print("\t开始计算SA策略综合梯度(SIM+版本SA包含D2A、VME、DIM、SIM、TIM)")
            grad = tf.zeros_like(x)
            for k in range(sample_num):
                # 进入第i+1个循环优化阶段的第j+1次迭代，第k+1个偏移样本梯度计算
                # 对样本实施D2A策略进行增强
                x_nes = transformation_d2a(adv_x)
                # 实施VME & SIM策略，集成点在logits计算后，见compute_gradient_ca2()内部实现
                #   由于SIM+版本还引入DIM策略，此处传递给compute_gradient_ca2()的模型为合入了DIM变换的model_fn_dim()
                grad = grad + compute_gradient_ca2(model_fn_dim, loss_fn, x_nes, y, targeted)
            grad = grad / sample_num

            # 计算引入TIM策略后梯度
            grad = tf.nn.depthwise_conv2d(grad, stack_kernel_tim, strides=[1, 1, 1, 1], padding='SAME')

            # 计算累积梯度
            red_ind = list(range(1, len(grad.shape)))
            avoid_zero_div = tf.cast(1e-12, grad.dtype)
            grad = grad / tf.math.maximum(
                avoid_zero_div,
                tf.math.reduce_mean(tf.math.abs(grad), red_ind, keepdims=True),
            )
            momentum = momentum_decay_factor * momentum + grad
            # 计算累积梯度 结束
            # 计算SA(D2A & VME)策略综合梯度 结束

            # 此处的cleverhans中的momentum_iterative_method实现与MIM原文伪代码不一致
            #   根据MIM原文伪代码 optimal_perturbation 应等于 步长 * sign(momentum)，CA2中的实现与原文伪代码一致
            #       但此处的optimize_linear()，其中针对不同范数，对momentum的应用不一样，来自cleverhans中的实现
            #       当norm==tf.inf时，optimize_linear()的行为与原文一致，所以使用optimize_linear()时，只使用norm==tf.inf场景
            #       当前迭代的步长为eps_iter：最大扰动范围 / 第i个循环优化阶段的迭代总数
            eps_iter = max_epsilon / phase_step[i]
            optimal_perturbation = optimize_linear(momentum, eps_iter, norm)

            # 更新对样样本
            adv_x = adv_x + optimal_perturbation

            # clip_eta()来自cleverhans中的实现，使用时，只使用norm==tf.inf场景
            adv_x = x + clip_eta(adv_x - x, norm, max_epsilon)

            if clip_min is not None and clip_max is not None:
                adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x


def transformation_d2a(x):
    """
    采用D2A策略对输入样本进行’增强‘变换
    :param x: 输入样本
    :return: 根据D2A变换后的增强样本
    """
    # CA2.py中由于使用inception模型，样本均归一化到(-1,1). sample_variance的配置也作用在归一化后的样本上，文章实验显示最优配置为0.1
    #   文中在(0,1)语境下表述，最优配置为0.05
    # 本代码中本地使用ResNet50模型，样本为(0,255)，因此在应用D2A策略相关配置时，将样本先归一化到(0,1)，计算完在还原至(0,255)
    x = x / 255.0

    # 随机采样得到偏移向量
    # 注意：如果全局没有配置特定的种子，则每次运行结果都将不同；若出于特定实验目的，可在全局配置中设置固定种子
    vector = tf.random.normal(shape=x.shape)

    # sample_variance为偏移距离（文章中的ω）
    x_nes = x + sample_variance * tf.sign(vector)

    x_nes = x_nes * 255.0
    return x_nes


def transformation_dim(x):
    """
    采用DIM策略对输入样本进行’增强‘变换
    :param x: 输入样本
    :return: 根据DIM变换后的增强样本
    """
    # 获取一个介于本地模型输入要的图像大小(image_resize)和图像原图大小(image_size)间的随机整数
    rnd = tf.random.uniform((), image_resize, image_size, dtype=tf.int32)
    # 获得随机数，后对输入样本进行随机resize变换
    rescaled = tf.image.resize(x, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = image_size - rnd
    w_rem = image_size - rnd
    pad_top = tf.random.uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random.uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((x.shape[0], image_size, image_size, 3))

    # 此处用到了DIM算法的关键参数prod
    #   随机数<prod，则使用ret=padded，即采用上面的resize&padded后的样本
    #   随机数>=prod，则使用原输入样本，ret=input_tensor
    ret = tf.cond(tf.random.uniform(shape=[1])[0] < tf.constant(diverse_probability), lambda: padded, lambda: x)
    # 在经过概率选择后，又进行了一次resize变换，重新调整会模型输入的要求大小
    ret = tf.image.resize(ret, [image_resize, image_resize], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return ret


@tf.function
def compute_gradient_ca2(model_fn, loss_fn, x_nes, y, targeted):
    """
    计算应用SA策略后的综合梯度
    :param model_fn: 本地模型函数，调用可返回输入样本的logits；当前仅支持攻击单个模型，暂不支持攻击集成的多个模型
    :param loss_fn: 损失函数，入参格式为(labels, logits) ，返回损失值
    :param x_nes: 输入样本，实施样本增强后的样本
    :param y: (可选) 当进行targeted攻击时，y设置为targeted标签（格式要求同：tf.argmax(model(images_target), 1)）
    :param targeted: (可选 bool) 当进行targeted攻击时，设置为True
    :return: 应用SA策略后的综合梯度
    """

    with tf.GradientTape() as g:
        g.watch(x_nes)

        # 初始化卷积核数量，也即VME策略虚拟模型数量
        stack_kernel_num = len(list_stack_kernel_size)
        base_logits = model_fn(x_nes)
        # 初始化logits为0
        logits = tf.zeros_like(base_logits)

        # 计算VME策略综合logits
        for i in range(stack_kernel_num):
            # 如果卷积核尺寸为1*1，则相当于不变，则无需真的进行卷积计算，直接累加原logits
            if list_stack_kernel_size[i] == 1:
                logits = logits + base_logits

            # 如卷积核尺寸不为1*1，则卷积计算后，再计算logits，并累加
            else:
                x_conv = tf.nn.depthwise_conv2d(x_nes, stack_kernel_list[i], strides=[1, 1, 1, 1], padding='SAME')
                logits = logits + model_fn(x_conv)

        # 计算SIM策略综合logits，共形成3个虚拟模型（分别使用1/2、1/4、1/8进行像素值缩放）
        for j in range(3):
            logits = logits + model_fn(x_nes * (1 / math.pow(2, j+1)))

        # 多个虚拟模型（VME & SIM策略）输出的均值，用于后续计算当前增强样本的综合损失函数
        logits = logits / (stack_kernel_num + 3)

        # 计算综合损失函数
        loss = loss_fn(labels=y, logits=logits)
        if targeted:
            loss = -loss

    grad = g.gradient(loss, x_nes)
    return grad


def loss_fn(labels, logits):
    """
    Added softmax cross entropy loss for MIM as in the original MI-FGSM paper.
    """

    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name=None)
