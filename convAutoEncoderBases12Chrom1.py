"""compress 11 PGP VCF files and JW VCF, chrom 1. genomes are:
        'hu6664F3',
        'huB1BE8C',
        'huD57BBF',
        'hu2C27FF',
        'hu816A0B',
        'huBF974E',
        'huDD583A',
        'JW',
        'hu365511',
        'huC3C30D',
        'huCF305F',
        'huhu7ECB9C
"""

import tensorflow as tf
import numpy as np
import math
from activations import lrelu
import os
#from libs.utils import corrupt
import time

GAPS_SEGMENT_LENGTH = 14400#14400 #must be pefect square 120x120
GAPS_SEGMENT_LENGTH_SR = 120 #120 #120 #square root
GAPS_COMPRESSION_RATIO = 64 #64 #(64*25) #64 #must be perfect square
GAPS_COMPRESSION_RATIO_SR = 8 #8 #(8*5)#8
LW_ROWS = GAPS_SEGMENT_LENGTH/GAPS_COMPRESSION_RATIO #decode layer's weight matrix rows, columns fix to SEGMENT LENGTH

LEARNING_RATE = 0.01
# %%
def autoencoder(input_shape=[None, GAPS_SEGMENT_LENGTH],
                n_filters=[1, 5, 5, 1], #10, 10, 10
                filter_sizes=[4, 4, 4, 4], #10, 10, 10
                corruption=False):
    """Build a deep denoising autoencoder w/ tied weights.
    Parameters
    ----------
    input_shape : list, optional
        Description
    n_filters : list, optional
        Description
    filter_sizes : list, optional
        Description
    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    Raises
    ------
    ValueError
        Description
    """
    # %%
    # input to the network
    x = tf.placeholder(
        tf.float32, input_shape, name='x')


    # %%
    # ensure 2-d is converted to square tensor.
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(
            x, [-1, x_dim, x_dim, n_filters[0]])
        print "img is 2d", x.get_shape(), tf.shape(x_tensor)
    elif len(x.get_shape()) == 4:
        x_tensor = x
        print "img is 4d", x.get_shape
    else:
        raise ValueError('Unsupported input dimensions')
    current_input = x_tensor

    # %%
    # Optionally apply denoising autoencoder
    if corruption:
        #current_input = corrupt(current_input)
        pass


    # %%
    # Build the encoder
    encoder = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        #b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        if layer_i != 2: #change here if add or remove layers
            output = tf.nn.sigmoid(
                tf.nn.conv2d(
                current_input, W, strides=[1, 1, 1, 1], padding='SAME'))
        else:
            output = tf.nn.sigmoid(
                tf.nn.conv2d(
                current_input, W, strides=[1,
                        GAPS_COMPRESSION_RATIO_SR, GAPS_COMPRESSION_RATIO_SR, 1], padding='SAME'))
            #tf.add(tf.nn.conv2d(
            #    current_input, W, strides=[1, 1, 1, 1], padding='SAME'), b))
        current_input = output

    # %%
    # store the latent representation
    z = current_input

    #use a linear layer for decoding
    reshape = tf.reshape(z, [-1, LW_ROWS]) #compressed to 225
    dim = reshape.get_shape()[1].value
    LWeights = tf.Variable(
                tf.random_uniform([
                dim,
                GAPS_SEGMENT_LENGTH],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))

    #biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    y = tf.nn.sigmoid(tf.matmul(reshape,LWeights))

    # cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - x))

    # %%
    return {'x': x, 'z': z, 'y': y, 'cost': cost, 'w': LWeights, 'e':encoder}

def getBasesData(chrom, filename, segmentLength=GAPS_SEGMENT_LENGTH):
    """
    Reads data file containing bases (1 gap per line)
    :param filename: file containing bases
    :param segmentLength: length of each segment
    :return: two numpy arrays, with the first being the data that fits cleanly into
        the segment length (size of num_segments x segmentLength),
        and the second being the remainder data that gets left over.
    """
    snpToNumbers = {'AC': 1.0, 'AG': 2.0, 'AT': 3.0, 'CA': 4.0, 'CG': 5.0, 'CT': 6.0,
                    'GA': 7.0, 'GC': 8.0, 'GT': 9.0, 'TA': 10.0, 'TC': 11.0, 'TG': 12.0}

    print "Reading data..."
    rawData = []

    totalBases = 0
    for i in [
        'hu6664F3',
        'huB1BE8C',
        'huD57BBF',
        'hu2C27FF',
        'hu816A0B',
        'huBF974E',
        'huDD583A',
        'JW',
        'hu365511',
        'huC3C30D',
        'huCF305F',
        'huhu7ECB9C']:

        filename = "/home/tester/data/"+i+"/bases2chr"+chrom
        if os.path.isfile(filename) != True:
            print "not a file, ", filename
            continue
        else:
            print filename
        f = open(filename)
        rawDataOneGenoAllSz = [x.strip() for x in f.readlines()]
        rawDataOneGeno = [snpToNumbers[x] for x in rawDataOneGenoAllSz if x in snpToNumbers]

        rawDataOneGenoRounded = rawDataOneGeno[:(len(rawDataOneGeno) - len(rawDataOneGeno)%segmentLength)]

        #print len(rawDataOneGeno), len(rawDataOneGenoRounded)
        totalBases += len(rawDataOneGenoAllSz)

        rawData.extend(rawDataOneGenoRounded)
        #print rawData

    print totalBases

    leftOverMatrixIndex = len(rawData) - (len(rawData) % (segmentLength))
    data = np.ndarray((0, segmentLength), dtype=np.float32)
    for segment in xrange(len(rawData[:leftOverMatrixIndex]) / segmentLength):
        data = np.vstack((data, np.asarray(rawData[(segment * segmentLength):((segment + 1) * segmentLength)])))
    print data.shape[0], data.shape[1]
    return data

def normalize(data):
    """
    Normalizes data to between 0 and 1
    :param data: data to be normalized
    :return: normalized data
    """
    #average = data.max()/2.0
    #print "average is in normalize: ", average
    #return (data-average)/average
    print data.max(), data.min()
    return (data - data.min()) / (data.max() - data.min())

def denormalize(data, minValue, maxValue, average):
    """
    Inverse of normalize()
    :param data: data to be denormalized
    :return:
    """
    #print "average in denorm and the decoded data before denom: ", average, data
    #return data*average+average
    print maxValue, minValue
    return (data * (maxValue - minValue)) + minValue

def findErrorMatrix(input, reconstructedInput):
    """
    Creates the error matrix used in decompression
    :param input: original matrix
    :param reconstructedInput: matrix, having  been reconstructed after compression
    :return: error matrix containing position and difference of each inconsistency between the two matrices
    """
    errorMatrix = [[]]
    errorMatrixOriginal = input - reconstructedInput
    for i in xrange(errorMatrixOriginal.shape[0]):
        for j in xrange(errorMatrixOriginal.shape[1]):
            if errorMatrixOriginal[i,j] != 0:
                errorMatrix.append([i, j, errorMatrixOriginal[i, j]])
    print errorMatrix[0]
    del(errorMatrix[0])
    return errorMatrix

def test_bases(chrom):
    """Test the convolutional autoencder using MNIST."""
    # %%
    import tensorflow as tf
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    import matplotlib.pyplot as plt

    # %%
    # load gaps as before
    bases = getBasesData(chrom, 'bases')
    norm_img = normalize(bases)
    ae = autoencoder()

    # %%
    learning_rate = LEARNING_RATE #was 0.007
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])
    start_time = time.time()

    # %%
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # %%
    # Fit all training data
    batch_size = bases.shape[0]
    n_epochs = 30000 #15000 60000 #55000  before this test
    for epoch_i in range(n_epochs):
        for batch_i in range(bases.shape[0] // batch_size):
            batch_xs = norm_img
            #print batch_xs
            train = np.array([img for img in batch_xs])
            sess.run(optimizer, feed_dict={ae['x']: train})
        if epoch_i%2500 == 0:
            print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))
        y_whendone = sess.run(ae['y'], feed_dict={ae['x']: train})
        z_whendone = sess.run(ae['z'], feed_dict={ae['x']: train})
        w_whendone = sess.run(ae['w'], feed_dict={ae['x']: train})
        eWeights_whendone = sess.run(ae['e'], feed_dict={ae['x']: train})
        cost_now = sess.run(ae['cost'], feed_dict={ae['x']: train})
        if cost_now < 300:
            print "cost reaches 80"
            break
    end_time = time.time()
    print "Autoencoder comprepss time in seconds: {0:.3f} ".format(end_time - start_time)
    #print "encoder weights", eWeights_whendone
    #for w in eWeights_whendone:
    #    print w.shape

    #print  "z shape", z_whendone.shape, z_whendone
    y_img = np.reshape(y_whendone, (-1, GAPS_SEGMENT_LENGTH ))

    recovered_bases = denormalize(y_img, bases.min(), bases.max(), 0) #last param not used
    x_dim = recovered_bases.shape[0]
    y_dim = recovered_bases.shape[1]
    #print x_dim, y_dim
    for i in range(x_dim):
        for j in range(y_dim):
            recovered_bases[i][j] = round(recovered_bases[i][j])

    #print recovered_gaps
    errorMatrix = findErrorMatrix(bases, recovered_bases)
    accuracy = 100 - (len(errorMatrix) / (float(x_dim*y_dim))) * 100.0

    #fix the errors
    for i in xrange(len(errorMatrix)):
        row, col, error = errorMatrix[i]
        recovered_bases[row, col] += error

    assert np.array_equal(bases[0], recovered_bases[0])

    print accuracy, len(errorMatrix), (x_dim*y_dim)
    cost_whendone = np.sum(np.square(y_img - norm_img))
    print cost_whendone
    np.save('encodedData.npy', np.asarray(z_whendone, dtype=np.float16))
    np.save('decoder.npy', np.asarray(w_whendone, dtype=np.float16))
    np.save('errorMatrix'+chrom+'.npy', np.asarray(errorMatrix, dtype=np.int8))

if __name__ == '__main__':
    for i in range(1, 23):
        chrom = str(i)
        test_bases(chrom)