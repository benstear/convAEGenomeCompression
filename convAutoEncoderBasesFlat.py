"""create a convolutional autoencoder w/ Tensorflow for SNP compression, and store Indels in a separate file for compression
this is a flat model, meaning SNPs are not converted to square images
"""

import tensorflow as tf
import numpy as np
import math
import os
#from libs.utils import corrupt
import time

GAPS_SEGMENT_LENGTH = 14400 # 14400
GAPS_COMPRESSION_RATIO = 64 # 64


LEARNING_RATE = 0.01 #was 0.01
# %%
def autoencoder(n_filters=[1, 5, 5, 1],  # 10, 10, 10
                filter_sizes=[4, 4, 4, 4],  # 10, 10, 10
                segmentLength=GAPS_SEGMENT_LENGTH,
                compressionRatio=GAPS_COMPRESSION_RATIO):

    input_shape = [None, 1, segmentLength, 1]
    x = tf.placeholder(
        tf.float32, input_shape, name='x')

    current_input = x  # x_tensor

    # %%
    # Build the encoder
    encoder = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        print "Layer i"
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(
            tf.random_uniform([1, filter_sizes[layer_i], n_input, n_output], -1.0 / math.sqrt(n_input),
                              1.0 / math.sqrt(n_input)))
        # b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        if layer_i == 2:  # change here if add or remove layers
            output = tf.nn.sigmoid(tf.nn.conv2d(current_input, W, strides=[1, 1, compressionRatio, 1], padding='SAME'))

        else:
            output = tf.nn.sigmoid(tf.nn.conv2d(current_input, W, strides=[1, 1, 1, 1], padding='SAME'))
        current_input = output
    # %%
    # store the latent representation
    z = current_input

    # use a linear layer for decoding
    reshape = tf.reshape(z, [-1, segmentLength / compressionRatio])
    dim = reshape.get_shape()[1].value
    LWeights = tf.Variable(tf.random_uniform([dim, segmentLength], -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))

    # biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    y = tf.nn.sigmoid(tf.matmul(reshape, LWeights))

    # cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - x[:, 0, :, 0]))

    return {'x': x, 'z': z, 'y': y, 'cost': cost, 'w': LWeights, 'e': encoder}

def getBasesData(filename, segmentLength=GAPS_SEGMENT_LENGTH):
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

    refLengths = []
    altLengths = []
    refalt = []

    print "Reading data..."
    rawData = []
    oldRawData = []

    totalBases = 0
    totalSNPs  = 0

    for i in range(1, 23):
        rawDataOneGeno = []
        oldRawDataOneGeno = []

        filename = "/home/tester/Downloads/jw/JWIndel/"+"verifychr" + str(i)

        f = open(filename)

        for x in f.readlines():
            y= x.strip()
            chrom, loc, snp_indel = y.split(',')
            ref, alt = snp_indel.split('/')
            if '-' in ref:
                #insertion, only need alr
                altLengths.append(len(alt))
                refalt_sub = [base for base in alt]
                refalt.extend(refalt_sub)
                entry = 13
            elif '-' in alt:
                #del, only need ref
                refLengths.append(len(ref))
                refalt_sub = [base for base in ref]
                refalt.extend(refalt_sub)
                entry = 14
            else:
                key = ref + alt
                if key in snpToNumbers:
                    entry = snpToNumbers[key]
                    #print entry
                    totalSNPs += 1


            rawDataOneGeno.extend([[entry]])
            oldRawDataOneGeno.extend([entry])
        rawDataOneGenoRounded = rawDataOneGeno[:(len(rawDataOneGeno) - len(rawDataOneGeno)%segmentLength)]
        oldRawDataOneGenoRounded = oldRawDataOneGeno[:(len(oldRawDataOneGeno) - len(oldRawDataOneGeno)%segmentLength)]


        print len(rawDataOneGeno), len(rawDataOneGenoRounded)
        totalBases += len(rawDataOneGeno)

        rawData.extend(rawDataOneGenoRounded)
        oldRawData.extend(oldRawDataOneGenoRounded)
        print totalBases, totalSNPs, float((totalSNPs*100.0)/((float)(totalBases)))


    print totalBases
    leftOverMatrixIndex = len(rawData) - (len(rawData) % (segmentLength))

    data = np.ndarray((0, 1, segmentLength, 1), dtype=np.float32)
    # data = np.ndarray((0, segmentLength), dtype=np.float32)
    for segment in xrange(len(rawData[:leftOverMatrixIndex]) / segmentLength):
        datapoint = [[[]]]
        datapoint[0][0] = np.asarray(rawData[(segment * segmentLength):((segment + 1) * segmentLength)])
        data = np.vstack((data, datapoint))

    oldData = np.ndarray((0, segmentLength), dtype=np.float32)
    for segment in xrange(len(oldRawData[:leftOverMatrixIndex]) / segmentLength):
        oldData = np.vstack((oldData, np.asarray(oldRawData[(segment * segmentLength):((segment + 1) * segmentLength)])))

    #dump some data
    count = 0
    for i in altLengths:
        print "Alt: ", i
        count += 1
        if count >= 10:
            break
    count = 0
    for i in refLengths:
        print "Ref: ", i
        count += 1
        if count >= 10:
            break
    #save refalt bases in a file
    # np.save('ref_alt.npy', np.asarray(refalt))
    # np.save('refLengths.npy', np.asarray(refLengths, dtype=np.int8))
    # np.save('altLengths.npy', np.asarray(altLengths, dtype=np.int8))
    #encodeIndels(refalt, bc)
    return data, oldData

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

def test_bases():

    import tensorflow as tf

    i = 0
    while os.path.exists('weights' + str(i)):
        i += 1
    directoryName = 'weights' + str(i)
    os.mkdir(directoryName)
    os.mkdir(directoryName + '/compressed')

    # %%
    # load gaps as before
    testBases, bases = getBasesData('bases')
    norm_img = normalize(testBases)
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
    n_epochs = 30000 #60000 #55000  before this test
    for epoch_i in range(n_epochs):
        for batch_i in range(bases.shape[0] // batch_size):
            batch_xs = norm_img
            #print batch_xs
            train = np.array([img for img in batch_xs])
            sess.run(optimizer, feed_dict={ae['x']: train})
        #return {'x': x, 'z': z, 'y': y, 'cost': cost, 'w': LWeights, 'e': encoder}

        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))
        whendone = sess.run(ae, feed_dict={ae['x']: train})
        # y_whendone = sess.run(ae['y'], feed_dict={ae['x']: train})
        # z_whendone = sess.run(ae['z'], feed_dict={ae['x']: train})
        # w_whendone = sess.run(ae['w'], feed_dict={ae['x']: train})
        # eWeights_whendone = sess.run(ae['e'], feed_dict={ae['x']: train})
        # cost_now = sess.run(ae['cost'], feed_dict={ae['x']: train})

        y_whendone = whendone['y']
        z_whendone = whendone['z']
        w_whendone = whendone['w']
        eWeights_whendone = whendone['e']
        cost_now = whendone['cost']
        if cost_now < 80:
            print "cost reaches 80"
            break
    end_time = time.time()
    print "Autoencoder comprepss time in seconds: {0:.3f} ".format(end_time - start_time)
    for e in eWeights_whendone:
        print e.shape

    #print  "z shape", z_whendone.shape, z_whendone
    y_img = np.reshape(y_whendone, (-1, GAPS_SEGMENT_LENGTH ))

    recovered_bases = denormalize(y_img, bases.min(), bases.max(), 0) #last param not used
    x_dim = recovered_bases.shape[0]
    y_dim = recovered_bases.shape[1]
    print x_dim, y_dim
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

    np.save(directoryName + '/compressed/encodedData.npy', np.asarray(z_whendone, dtype=np.float16))
    np.save(directoryName + '/compressed/decoder.npy', np.asarray(w_whendone, dtype=np.float16))
    np.save(directoryName + '/compressed/errorMatrix.npy', np.asarray(errorMatrix, dtype=np.int8))
    i = 0
    for eweight in eWeights_whendone:
        np.save(directoryName + '/encoder' + str(i), np.asarray(eweight, dtype=np.float16))
        i += 1
    #np.save('encoder.npy', np.asarray(eWeights_whendone, dtype=np.float16))
    #print gaps

    # %%
    # Plot example reconstructions

    #recon = sess.run(ae['y'], feed_dict={ae['x']: norm_img})

    #print norm_img
    #print "recon", recon.shape
    #print recon



if __name__ == '__main__':
    test_bases()
