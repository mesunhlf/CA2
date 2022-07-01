# coding: utf-8
import os
import numpy as np
from PIL import Image
from scipy.misc import imread, imsave
import tensorflow as tf


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.
    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images

def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')


def images_to_FD(input_tensor):
    """Process the image to meet the input requirements of FD"""
    ret = tf.image.resize_images(input_tensor, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    ret = tf.reverse(ret, axis=[-1])  # RGB to BGR
    ret = tf.transpose(ret, [0, 3, 1, 2])
    return ret

def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern]).swapaxes(0, 2)
    stack_kern = np.expand_dims(stack_kern, 3)
    return stack_kern, kern_size // 2


def project_noise(x, stack_kern, kern_size):
    x = tf.pad(x, [[0,0],[kern_size,kern_size],[kern_size,kern_size],[0,0]], "CONSTANT")
    x = tf.nn.depthwise_conv2d(x, stack_kern, strides=[1, 1, 1, 1], padding='VALID')
    # x = tf.nn.depthwise_conv2d(x, stack_kern, strides=[1, 1, 1, 1], padding='SAME')
    return x


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3)
    return stack_kernel

def lkern():
    stack_kernel = 1
    return stack_kernel

def input_diversity(FLAGS, input_tensor):
    """Input diversity: https://arxiv.org/abs/1803.06978"""
    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    ret = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
    ret = tf.image.resize_images(ret, [FLAGS.image_height, FLAGS.image_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    if(FLAGS.use_dim):
        return ret
    else:
        return input_tensor



# coding: utf-8
import os
import numpy as np
from scipy.misc import imresize,imread,imsave
from PIL import Image
import tensorflow as tf
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2, resnet_v1, vgg, nets_factory

slim = tf.contrib.slim


def vgg_normalization(image):
  return image - [123.68, 116.78, 103.94]

def inception_normalization(image):
  return ((image / 255.) - 0.5) * 2

def inv_vgg_normalization(image):
  return np.clip(image + [123.68, 116.78, 103.94],0,255)

def inv_inception_normalization(image):
  return np.clip((image + 1.0) * 0.5 * 255,0,255)

normalization_fn_map = {
    'inception_v1': inception_normalization,
    'inception_v2': inception_normalization,
    'inception_v3': inception_normalization,
    'inception_v4': inception_normalization,
    'inception_resnet_v2': inception_normalization,
    'resnet_v1_50': vgg_normalization,
    'resnet_v1_101': vgg_normalization,
    'resnet_v1_152': vgg_normalization,
    'resnet_v1_200': vgg_normalization,
    'resnet_v2_50': inception_normalization,
    'resnet_v2_101': inception_normalization,
    'resnet_v2_152': inception_normalization,
    'resnet_v2_200': inception_normalization,
    'vgg_16': vgg_normalization,
    'vgg_19': vgg_normalization,
}

inv_normalization_fn_map = {
    'inception_v1': inv_inception_normalization,
    'inception_v2': inv_inception_normalization,
    'inception_v3': inv_inception_normalization,
    'inception_v4': inv_inception_normalization,
    'inception_resnet_v2': inv_inception_normalization,
    'resnet_v1_50': inv_vgg_normalization,
    'resnet_v1_101': inv_vgg_normalization,
    'resnet_v1_152': inv_vgg_normalization,
    'resnet_v1_200': inv_vgg_normalization,
    'resnet_v2_50': inv_inception_normalization,
    'resnet_v2_101': inv_inception_normalization,
    'resnet_v2_152': inv_inception_normalization,
    'resnet_v2_200': inv_inception_normalization,
    'vgg_16': inv_vgg_normalization,
    'vgg_19': inv_vgg_normalization,
}

offset = {
    'inception_v1': 1,
    'inception_v2': 1,
    'inception_v3': 1,
    'inception_v4': 1,
    'inception_resnet_v2': 1,
    'resnet_v1_50': 0,
    'resnet_v1_101': 0,
    'resnet_v1_152': 0,
    'resnet_v1_200': 0,
    'resnet_v2_50': 1,
    'resnet_v2_101': 1,
    'resnet_v2_152': 1,
    'resnet_v2_200': 1,
    'vgg_16': 0,
    'vgg_19': 0,
  }

image_size={
    'inception_v1': 299,
    'inception_v2': 299,
    'inception_v3': 299,
    'inception_v4': 299,
    'inception_resnet_v2': 299,
    'resnet_v1_50': 224,
    'resnet_v1_101': 224,
    'resnet_v1_152': 224,
    'resnet_v1_200': 224,
    'resnet_v2_50': 299,
    'resnet_v2_101': 299,
    'resnet_v2_152': 299,
    'resnet_v2_200': 299,
    'vgg_16': 224,
    'vgg_19': 224,
  }

base_path='/nfs/checkpoints'

checkpoint_paths = {
    'inception_v1': None,
    'inception_v2': None,
    'inception_v3': base_path+'/inception_v3.ckpt',
    'inception_v4': base_path+'/inception_v4.ckpt',
    'inception_resnet_v2': base_path+'/inception_resnet_v2_2016_08_30.ckpt',
    'resnet_v1_50': base_path+'/resnet_v1_50.ckpt',
    'resnet_v1_101': None,
    'resnet_v1_152': base_path+'/resnet_v1_152.ckpt',
    'resnet_v1_200': None,
    'resnet_v2_50': base_path+'/resnet_v2_50.ckpt',
    'resnet_v2_101': None,
    'resnet_v2_152': base_path+'/resnet_v2_152.ckpt',
    'resnet_v2_200': None,
    'vgg_16': base_path+'/vgg_16.ckpt',
    'vgg_19': base_path+'/vgg_19.ckpt',
    'adv_inception_v3':base_path+'/adv_inception/adv_inception_v3.ckpt',
    'adv_inception_resnet_v2':base_path+'/adv_inception_resnet_v2/adv_inception_resnet_v2.ckpt',
    'ens3_adv_inception_v3':base_path+'/ens3_adv_inception_v3/ens3_adv_inception_v3.ckpt',
    'ens4_adv_inception_v3':base_path+'/ens4_adv_inception_v3/ens4_adv_inception_v3.ckpt',
    'ens_adv_inception_resnet_v2':base_path+'/ens_adv_inception_resnet_v2/ens_adv_inception_resnet_v2.ckpt'
  }

ground_truth=None
with open('/nfs/dataset/ali2019/val.txt') as f:
    ground_truth=f.read().split('\n')[:-1]

def load_image(image_path, image_size, batch_size):
    images = []
    filenames=[]
    labels=[]
    idx=0
    files=os.listdir(image_path)
    files.sort(key=lambda x: str(x[:-4]))
    for i,filename in enumerate(files):
        # image = imread(image_path + filename)
        # image = imresize(image, (image_size, image_size)).astype(np.float)
        image=Image.open(image_path + filename)
        image=image.resize((image_size,image_size))
        image=np.array(image)
        images.append(image)
        filenames.append(filename)
        labels.append(int(ground_truth[i][-3:]))
        idx+=1
        if idx==batch_size:
            yield np.array(images),np.array(filenames),np.array(labels)
            idx=0
            images=[]
            filenames=[]
            labels=[]
    if idx>0:
        yield np.array(images), np.array(filenames),np.array(labels)

def save_image(images,names,output_dir):
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i, name in enumerate(names):
        # imsave(output_dir+name,images[i].astype('uint8'))
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)




if __name__=='__main__':
    pass

