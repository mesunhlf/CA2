"""
Tensorflow 2.0版本的CA2基础版实现，与原TF1版的的CA2.py对应

经过测试的TensorFlow版本：2.7.0 、 2.8.0

@Skydiver
"""

# 导入全局配置，最大噪声范围、RO/SA策略配置等
from conf_basic import *
from utils_tf2 import *


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
    Tensorflow 2.0版本的CA2基础版实现，与原TF1版的的CA2.py对应
    该实现中基于动量的优化部分参考cleverhans中的momentum_iterative_method. 调整的部分，均基于尊重MIM和CA2原文实现
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

    # 初始化循环优化动量/梯度
    momentum = tf.zeros_like(x)
    adv_x = tf.zeros_like(x)

    for i in range(phase_num):
        # 进入第i+1个循环优化阶段
        print("---- 进入第%d个循环优化阶段, 当前阶段迭代总数为%d ----" % (i + 1, phase_step[i]))

        # 每个循环优化阶段开始时，样本均初始化为原始样本（即重新出发）
        adv_x = x
        # 根据循环优化策略，动量初始化为上一阶段循环优化动量 * 学习权重
        momentum = momentum * momentum_learn_factor

        for j in range(phase_step[i]):
            # 进入第i+1个循环优化阶段的第j+1次迭代
            print("--- 进入第%d个循环优化阶段的第%d次迭代 ---" % (i + 1, j + 1))

            # 计算SA(D2A & VME)策略综合梯度
            print("\t开始计算SA策略综合梯度(基础版本SA仅包含D2A、VME)")
            grad = tf.zeros_like(x)
            for k in range(sample_num):
                # 进入第i+1个循环优化阶段的第j+1次迭代，第k+1个偏移样本梯度计算

                # 对样本实施D2A策略进行增强
                x_nes = transformation_d2a(adv_x)
                # VME策略，集成点在logits计算后，见compute_gradient_ca2()内部实现
                grad = grad + compute_gradient_ca2(model_fn, loss_fn, x_nes, y, targeted)
            grad = grad / sample_num

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
    #   实则文中在(0,1)语境下表述，最优配置为0.05
    # 本代码中本地使用ResNet50模型，样本为(0,255)，因此在应用D2A策略相关配置时，将样本先归一化到(0,1)，计算完在还原至(0,255)
    x = x / 255.0

    # 随机采样得到偏移向量
    # 注意：如果全局没有配置特定的种子，则每次运行结果都将不同；若出于特定实验目的，可在全局配置中设置固定种子
    vector = tf.random.normal(shape=x.shape)

    # sample_variance为偏移距离（文章中的ω）
    x_nes = x + sample_variance * tf.sign(vector)

    x_nes = x_nes * 255.0
    return x_nes


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

        # 初始化卷积核数量
        stack_kernel_num = len(list_stack_kernel_size)
        base_logits = model_fn(x_nes)
        # 初始化logits为0
        logits = tf.zeros_like(base_logits)

        for i in range(stack_kernel_num):

            # 如果卷积核尺寸为1*1，则相当于不变，则无需真的进行卷积计算，直接累加原logits
            if list_stack_kernel_size[i] == 1:
                logits = logits + base_logits

            # 如卷积核尺寸不为1*1，则卷积计算后，再计算logits，并累加
            else:
                x_conv = tf.nn.depthwise_conv2d(x_nes, stack_kernel_list[i], strides=[1, 1, 1, 1], padding='SAME')
                logits = logits + model_fn(x_conv)

        # 多个虚拟模型输出的均值，用于后续计算当前增强样本的综合损失函数
        logits = logits / stack_kernel_num

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
