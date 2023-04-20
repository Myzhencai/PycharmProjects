# --coding:utf-8--
# 获得模型信息的代码
# https://blog.csdn.net/m0_37935211/article/details/83021723
# https://codeantenna.com/a/UOS8pF1rCV
# from keras.applications.densenet import DenseNet201, preprocess_input
# from keras.layers import Dense, GlobalAveragePooling2D
# from keras.models import Model
#
# # base_model = DenseNet(weights='imagenet', include_top=False)
# base_model = DenseNet201(include_top=False)
#
# inputshape = base_model.input
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# predictions = Dense(5, activation='softmax')(x)
# model = Model(inputs=base_model.input, outputs=predictions)
#
# model.summary()
# print('the number of layers in this model:' + str(len(model.layers)))
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D, Dense, Dropout
from keras.layers import Dense, Add
from keras.layers import concatenate, LSTM
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input, Flatten
from keras.layers import MaxPooling2D
from keras.optimizers import Adam
import os
import matplotlib.pyplot as plt
from keras import regularizers, optimizers
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import cv2
from sklearn.model_selection import train_test_split
import keras
import serial
import  time

def dense(input):
    # input = Input(shape=(128, 128, 3))
    conv1a = Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(input)
    # conv1a = BatchNormalization()(conv1a)
    conv1b = Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(conv1a)
    # conv1b = BatchNormalization()(conv1b)
    merge1 = concatenate([conv1a, conv1b], axis=-1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(merge1)

    conv2a = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
    # conv2a = BatchNormalization()(conv2a)
    conv2b = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(conv2a)
    # conv2b = BatchNormalization()(conv2b)
    merge2 = concatenate([conv2a, conv2b], axis=-1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(merge2)

    conv3a = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
    # conv3a = BatchNormalization()(conv3a)
    conv3b = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(conv3a)
    # conv3b = BatchNormalization()(conv3b)
    merge3 = concatenate([conv3a, conv3b], axis=-1)
    pool3 = MaxPooling2D(pool_size=(2, 2))(merge3)

    conv4a = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(pool3)
    # conv4a = BatchNormalization()(conv4a)
    conv4b = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(conv4a)
    # conv4b = BatchNormalization()(conv4b)
    merge4 = concatenate([conv4a, conv4b], axis=-1)
    pool4 = MaxPooling2D(pool_size=(1, 1))(merge4)

    flatten = Flatten()(pool4)
    # flatten = Flatten()(merge4)
    dense1 = Dense(128, activation='sigmoid')(flatten)
    dense2 = Dropout(0.25)(dense1)
    output = Dense(4, activation='softmax')(dense2)

    model = Model(inputs=input, outputs=output)
    return model

def loaddata(filepath):
    all_data = np.loadtxt(filepath)
    return all_data

# ------------------------------定义参数------------------------------
TIME_STEPS = 1  # 时间点数据 每次读取1行共28次 same as the height of the image
INPUT_SIZE = 9  # 每行读取28个像素点 same as the width of the image
BATCH_SIZE = 128 # 每个批次训练50张图片
BATCH_INDEX = 0
OUTPUT_SIZE = 4  # 每张图片输出分类矩阵
CELL_SIZE = 128  # RNN中隐藏单元
LR = 0.001  # 学习率

# ------------------------------數據加載------------------------------
path = "/home/gaofei/PycharmProjects/ElectroMagnetArea/SoarFacedata/megedData.txt"
# path = "/home/gaofei/PycharmProjects/ElectroMagnetArea/SoarFacedata7new/megedData.txt"
dataSet = loaddata(path)
x = dataSet[:,:9]
y = dataSet[:,9:13]


# 轉換數據的大小格式
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.25)
X_train = X_train.reshape((X_train.shape[0],1,X_train.shape[1]))
Xtrain_trans = np.zeros((X_train.shape[0],9,9))
Xtest_trans = np.zeros((X_test.shape[0],9,9))

for i in range(X_train.shape[0]):
    transX_train = np.tile(X_train[i][0], (9, 1))
    Xtrain_trans[i] = transX_train

X_test = X_test.reshape((X_test.shape[0],1,X_test.shape[1]))
for i in range(X_test.shape[0]):
    transX_test = np.tile(X_test[i][0], (9, 1))
    Xtest_trans[i] = transX_test

Xtrain_trans = Xtrain_trans.reshape((-1,9,9,1))
Xtest_trans = Xtest_trans.reshape((-1,9,9,1))


# ------------------------------創建模型------------------------------
input = Input(shape=(9, 9 ,1))
model = dense(input)

# ------------------------------確定優化器------------------------------
adam = Adam(LR)
model.compile(optimizer=adam,  # 加速神经网络
              loss='categorical_crossentropy',  # 损失函数
              metrics=['accuracy'])  # 计算误差或准确率


X_batch_trans = np.zeros((BATCH_SIZE,9,9))
# print(model.summary())
# ------------------------------訓練模型------------------------------
for step in range(8000):
    # 分批截取数据 BATCH_INDEX初始值为0 BATCH_SIZE为50 取28个步长和28个INPUT_SIZE
    X_batch = Xtrain_trans[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :, : , :]
    # Y_batch = y_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :].reshape((-1,1,4))
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :]
    # 计算误差
    cost = model.train_on_batch(X_batch, Y_batch)

    # 累加参数
    BATCH_INDEX += BATCH_SIZE
    # 如果BATCH_INDEX累加大于总体的个数 则重新赋值0开始分批计算
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    # # 每隔200步输出
    if step % 200 == 0:
        # 评价算法
        cost, accuracy = model.evaluate(
            Xtest_trans, y_test,
            batch_size=y_test.shape[0],
            verbose=1)
        print('test cost: ', cost, 'test accuracy: ', accuracy)

# lefteye = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/lefteye.png")
# leftface = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/leftface.png")
# lefthead = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/lefthead.png")
# leftjaw = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/leftjaw.png")
#
# ser = serial.Serial("/dev/ttyUSB0",256000,timeout = 0.01) # 开启com3口，波特率115200，超时5
# ser.flushInput() # 清空缓冲区
# while True:
#     currentdata = ser.readline() # 获取串口缓冲区数据
#     if currentdata !=b'' :
#         starttime= time.time()
#         currentdata = str(currentdata, 'UTF-8')
#         currentdatalist = currentdata.split('\r\n')[0]
#         currentdatalist = currentdatalist.split(",")
#         dataarray = np.array(currentdatalist,dtype='float32').reshape((-1,9))
#         # input = dataarray.reshape((1, 1, 9))
#         # 變爲991
#         input= np.tile(dataarray[0], (9, 1))
#         input = input.reshape((1,9,9,1))
#         result = model.predict(input)
#         result = result[0]
#         pred_y = np.argmax(result, 0)
#         endtime = time.time()
#         print("單次運算的時間:",endtime-starttime)
#         if pred_y==0:
#             print("當前在額頭區域")
#             cv2.imshow('image', lefthead)
#             cv2.waitKey(30)
#         elif pred_y==1:
#             # print("current is" ,areaid)
#             print("當前在下頜線區域")
#             cv2.imshow('image', leftjaw)
#             cv2.waitKey(30)
#         elif pred_y==2:
#             # print("current is" ,areaid)
#             print("當前在面部區域")
#             cv2.imshow('image', leftface)
#             cv2.waitKey(30)
#         elif pred_y==3:
#             # print("current is" ,areaid)
#             print("當前在眼周區域")
#             cv2.imshow('image', lefteye)
#             cv2.waitKey(30)

# 保存模型
model.save('KerasDensenet.h5')
newmodel = keras.models.load_model('KerasDensenet.h5')
print(newmodel.summary())
