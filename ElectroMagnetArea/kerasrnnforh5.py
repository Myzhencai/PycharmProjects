# -*- coding: utf-8 -*-
import keras
import numpy as np
np.random.seed(1337)
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam
from tensorflow.keras.models import load_model
import h5py
TIME_STEPS = 28
INPUT_SIZE = 28
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 10
CELL_SIZE = 50
LR = 0.001


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28) / 255
x_test = x_test.reshape(-1, 28, 28) / 255
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


model = Sequential()
model.add(SimpleRNN(
    units=CELL_SIZE,
    input_shape=(TIME_STEPS, INPUT_SIZE),

))
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

adam = Adam(LR)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


for step in range(4001):

    x_batch = x_train[BATCH_INDEX:BATCH_SIZE+BATCH_INDEX, :, :]
    y_batch = y_train[BATCH_INDEX:BATCH_SIZE+BATCH_INDEX, :]

    cost = model.train_on_batch(x_batch, y_batch)

    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= x_train.shape[0] else BATCH_INDEX


    if step % 500 == 0:
        cost, accuracy = model.evaluate(x_test, y_test, batch_size=y_test.shape[0])
        print(cost, accuracy)
# https://blog.csdn.net/u010879745/article/details/107944932
model.save('/home/gaofei/PycharmProjects/ElectroMagnetArea/modelfile/KerasRnnmodel.h5')

newmodel = load_model('/home/gaofei/PycharmProjects/ElectroMagnetArea/modelfile/KerasRnnmodel.h5')
print(newmodel.summary())

# f = h5py.File("/home/gaofei/PycharmProjects/ElectroMagnetArea/modelfile/KerasRnnmodel.h5", "w")