import os
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Flatten, Dropout
from keras.optimizers import Adam, SGD
# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF

WEIGHTS_FILE = 'weights_twostate.h5'

#enable JIT
# config = tf.ConfigProto()
# config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
# session = tf.Session(config=config)
# KTF.set_session(session)

model = Sequential()

model.add(Dense(output_dim=100, input_shape=(2, 8, 8, 12,)))
model.add(Flatten())
# model.add(Dropout(0.1))
model.add(Activation("relu"))
model.add(Dense(output_dim=4000))
model.add(Activation("relu"))
model.add(Dense(output_dim=2000))
model.add(Activation("relu"))
model.add(Dense(output_dim=2000))
# model.add(Dropout(0.1))
model.add(Activation("relu"))
model.add(Dense(output_dim=1000))
model.add(Activation("relu"))
model.add(Dense(output_dim=500))
model.add(Activation("relu"))
model.add(Dense(output_dim=100))
# model.add(Dropout(0.1))
model.add(Activation("relu"))
model.add(Dense(output_dim=2))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer=SGD(momentum=0.5))

if os.path.isfile(WEIGHTS_FILE):
    model.load_weights(WEIGHTS_FILE)
