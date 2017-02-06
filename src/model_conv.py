from keras.models import Sequential
from keras.layers import Dense, Activation, Merge
from keras.layers.core import Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD
import os.path

WEIGHTS_FILE = 'weights_conv.h5'

model = Sequential()
model.add(Convolution2D(258, 8, 8, input_shape=(8,8,12,), border_mode='same'))
model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(output_dim=2048))
model.add(Dropout(0.1))
model.add(Activation("relu"))
model.add(Dense(output_dim=4096))
model.add(Dropout(0.1))
model.add(Activation("relu"))
model.add(Dense(output_dim=1024))
model.add(Dropout(0.1))
model.add(Activation("relu"))
model.add(Dense(output_dim=2))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

if os.path.isfile(WEIGHTS_FILE):
	model.load_weights(WEIGHTS_FILE)