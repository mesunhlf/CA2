from datetime import datetime
from utils_tf2 import *

is_plus = True
if is_plus:
    import CA2_SIM_TF2
else:
    import CA2_TF2


def ca2_tf2_demo():

    print("-- 初始化模型: ResNet50 --")
    model = tf.keras.applications.ResNet50(weights="imagenet")
    bounds = (0, 255)

    print("-- 读取攻击目标图像 --")
    paths = ["images/imagenet_06_609.jpg", "images/imagenet_01_559.jpg"]
    images = samples(bounds=bounds, batchsize=1, index=1, paths=paths, shape=(224, 224))

    original_label = model(images)
    print("使用本地模型预测的目标图像的top标签: ", np.argmax(original_label))
    print('使用本地模型预测的目标图像的top分类:',
          tf.keras.applications.resnet50.decode_predictions(model.predict(images))[0])

    # images_target, labels_target = ref.use_define_samples.samples(fmodel=None, kmodel=model, dataset='imagenet',
    #                                                               bounds=bounds, batchsize=1, index=1, paths=paths)
    # print("使用LocalModel预测的Target图像的top标签: ", np.argmax(model(images_target)))

    print("-- 开始攻击 --")
    adv_x = CA2_TF2.ca2_tf2(model_fn=model, x=images)

    adv_x_label = model(adv_x)
    print("-- 攻击结束 --")
    print("使用本地模型预测的adv图像的top标签: ", np.argmax(adv_x_label, 1))

    if is_adv(original_label, adv_x_label):
        check_or_create_dir("output")
        filename = "output/adv_ca2_basic_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".jpg"
        print(
            "\n---- 攻击得出的数字对抗样本的top标签 %d ，保存为文件: %s ----" % (int(np.argmax(adv_x_label)), filename))
        save_image(adv_x[0], filename)
        images = samples(bounds=bounds, batchsize=1, index=0, paths=[filename], shape=(224, 224))
        adv_x_label_save = model(images)
        print("使用本地模型预测的保存的adv图像的top标签: ", np.argmax(adv_x_label_save))
        print('使用本地模型预测的保存的adv图像的top分类:',
              tf.keras.applications.resnet50.decode_predictions(model.predict(images))[0])


def ca2_sim_tf2_demo():

    print("-- 初始化模型: ResNet50 --")
    model = tf.keras.applications.ResNet50(weights="imagenet")
    bounds = (0, 255)

    print("-- 读取攻击目标图像 --")
    paths = ["images/imagenet_06_609.jpg", "images/imagenet_01_559.jpg"]
    images = samples(bounds=bounds, batchsize=1, index=1, paths=paths, shape=(224, 224))

    original_label = model(images)
    print("使用本地模型预测的目标图像的top标签: ", np.argmax(original_label))
    print('使用本地模型预测的目标图像的top分类:',
          tf.keras.applications.resnet50.decode_predictions(model.predict(images))[0])

    # images_target, labels_target = ref.use_define_samples.samples(fmodel=None, kmodel=model, dataset='imagenet',
    #                                                               bounds=bounds, batchsize=1, index=1, paths=paths)
    # print("使用LocalModel预测的Target图像的top标签: ", np.argmax(model(images_target)))

    print("-- 开始攻击 --")
    adv_x = CA2_SIM_TF2.ca2_tf2(model_fn=model, x=images)

    adv_x_label = model(adv_x)
    print("-- 攻击结束 --")
    print("使用本地模型预测的adv图像的top标签: ", np.argmax(adv_x_label, 1))

    if is_adv(original_label, adv_x_label):
        check_or_create_dir("output")
        filename = "output/adv_ca2_sim_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".jpg"
        print(
            "\n---- 攻击得出的数字对抗样本的top标签 %d ，保存为文件: %s ----" % (int(np.argmax(adv_x_label)), filename))
        save_image(adv_x[0], filename)
        images = samples(bounds=bounds, batchsize=1, index=0, paths=[filename], shape=(224, 224))
        adv_x_label_save = model(images)
        print("使用本地模型预测的保存的adv图像的top标签: ", np.argmax(adv_x_label_save))
        print('使用本地模型预测的保存的adv图像的top分类:',
              tf.keras.applications.resnet50.decode_predictions(model.predict(images))[0])


if __name__ == "__main__":

    if is_plus:
        print("*********测试PLUS版本*********")
        ca2_sim_tf2_demo()
    else:
        print("*********测试BASIC版本*********")
        ca2_tf2_demo()

