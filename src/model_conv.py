from keras.models import Sequential, Model
from keras.layers import Dense, Activation, merge, Input
from keras.layers.core import Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam, SGD
import os.path
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras import initializations
# from keras.utils.visualize_util import plot

#enable JIT
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
session = tf.Session(config=config)
KTF.set_session(session)

WEIGHTS_FILE = 'weights_conv.h5'

inp = Input(shape=(8,8,12,))

def my_init(shape, name=None, dim_ordering=None):
    return initializations.normal(shape, scale=0.01, name=name)

flatten = Flatten()

conv1 = Convolution2D(64, 3, 3, border_mode='valid', activation='relu', init=my_init)
conv1_out = conv1(inp)
conv1_out = flatten(conv1_out)

conv2 = Convolution2D(128, 5, 5, border_mode='valid', activation='relu', init=my_init)
conv2_out = conv2(inp)
conv2_out = flatten(conv2_out)

conv3 = Convolution2D(64, 8, 1, border_mode='valid', activation='relu', init=my_init)
conv3_out = conv3(inp)
conv3_out = flatten(conv3_out)

conv4 = Convolution2D(64, 1, 8, border_mode='valid', activation='relu', init=my_init)
conv4_out = conv4(inp)
conv4_out = flatten(conv4_out)

dense = Dense(512, activation='relu', init=my_init)
dense_out = dense(inp)
dense_out = flatten(dense_out)

out = merge([conv1_out, conv2_out, conv3_out, conv4_out, dense_out], mode='concat')

# out = Dropout(0.01)(out)
d1 = Dense(2048, activation='relu')
out = d1(out)
# out = Dropout(0.1)(out)
d2 = Dense(1024, activation='relu')
out = d2(out)
# out = Dropout(0.1)(out)
d3 = Dense(2, activation='relu')
out = d3(out)
out = Activation("softmax")(out)

model = Model(input=inp, output=out)
model.compile(loss='categorical_crossentropy', optimizer=SGD(momentum=0.5))

if os.path.isfile(WEIGHTS_FILE):
	model.load_weights(WEIGHTS_FILE)

# plot(model, to_file='model.png')
# plot(model, show_shapes=True, to_file='model_shapes.png')