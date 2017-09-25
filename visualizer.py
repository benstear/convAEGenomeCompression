from convAutoEncoderBases import getBasesData, normalize, denormalize

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

segmentLength = 14400
compressionRatio = 64


W_1 = np.load('firstWorking/encoder0.npy') #replace these paths with the actual paths to the encoders
W_2 = np.load('firstWorking/encoder1.npy')
input_shape = [None, 1, segmentLength, 1]
x = tf.placeholder(tf.float32, input_shape, name='x')

encode_1 = tf.nn.sigmoid(tf.nn.conv2d(x, W_1, strides=[1, 1, 1, 1], padding='SAME'))
encode_2 = tf.nn.sigmoid(tf.nn.conv2d(encode_1, W_2, strides=[1, 1, 1, 1], padding='SAME'))

sess = tf.Session()

def plotNNFilter(units, stimuli):

    filters = units.shape[3]
    img = np.vstack((np.tile(stimuli[0, :, :, 0], [300, 1]), np.tile(units[0, :, :, 0], [300, 1])))
    for i in range(1, filters):
        img = np.vstack((img, np.tile(units[0, :, :, i], [300, 1])))
    print img.shape
    plt.imshow(img)
    plt.show()


def getActivations(layer, stimuli):
    units = sess.run(layer,feed_dict={x:stimuli})
    plotNNFilter(units, stimuli)

bases = getBasesData('bases')[0]
bases = normalize(bases)

test = bases[[0], :, :, :]
plt.imshow(np.tile(test[0, :, :, 0], [300, 1]))
# plt.show()

getActivations(encode_1, test)
getActivations(encode_2, test)
