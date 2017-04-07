from keras.models import Sequential, Model
from keras.layers import Dense, Activation, merge, Input
from keras.layers.core import Flatten, Dropout
from keras.layers.convolutional import Convolution2D, Conv3D
from keras.optimizers import Adam, SGD
import os.path
import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
from keras import initializations
# from keras.utils.visualize_util import plot

# #enable JIT
# config = tf.ConfigProto()
# config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
# session = tf.Session(config=config)
# KTF.set_session(session)

WEIGHTS_FILE = 'weights_3dconv.h5'

inp = Input(shape=(2,8,8,12,))


conv1 = Conv3D(64, (2, 2, 2), border_mode='valid', activation='relu')
conv1_out = conv1(inp)
print(conv1.output_shape)
conv2 = Conv3D(64, (2, 3, 3), border_mode='valid', activation='relu')
conv2_out = conv2(inp)
print(conv2.output_shape)
conv3 = Conv3D(128, (2, 5, 5), border_mode='valid', activation='relu')
conv3_out = conv3(inp)
print(conv3.output_shape)
conv4 = Conv3D(128, (2, 8, 8), border_mode='valid', activation='relu')
conv4_out = conv4(inp)
print(conv4.output_shape)
# dense = Dense(128, activation='relu', init=my_init)
# dense_out = dense(inp)

out = merge([conv1_out, conv2_out, conv3_out, dense_out], mode='concat')

out = Flatten()(out)
# out = Dropout(0.01)(out)
d1 = Dense(2048, activation='relu', init=my_init)
out = d1(out)
# out = Dropout(0.1)(out)
d2 = Dense(4096, activation='relu', init=my_init)
out = d2(out)
# out = Dropout(0.1)(out)
d3 = Dense(2, activation='relu', init=my_init)
out = d3(out)
out = Activation("softmax")(out)

model = Model(input=inp, output=out)
model.compile(loss='categorical_crossentropy', optimizer=SGD())

if os.path.isfile(WEIGHTS_FILE):
	model.load_weights(WEIGHTS_FILE)

# plot(model, to_file='model.png')
# plot(model, show_shapes=True, to_file='model_shapes.png')