from vgg import vgg16
import tensorflow as tf
import numpy as np
from skimage.util import img_as_ubyte
from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt
from imagenet_classes import class_names

from scipy.misc import imread, imresize

print("Loading image")
img_path = 'laska.png'
img = imread('laska.png', mode='RGB')
img = imresize(img, (224, 224))
# img = io.imread(img_path)
# img = resize(img, (224,224))
x = np.expand_dims(img, axis=0)
x = x[:,:,:,::-1]

sess = tf.Session()

print("Loading Vgg")
imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

print("Feedforwarding")
prob = sess.run(vgg.probs, feed_dict={vgg.imgs: x})[0]
preds = (np.argsort(prob)[::-1])[0:5]
for p in preds:
    print(class_names[p], prob[p])

