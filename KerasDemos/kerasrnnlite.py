
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 18:43:06 2020
@author: xiuzhang Eastmount CSDN
Wuhan fighting!
"""
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import adam_v2
import keras
from sklearn.model_selection import train_test_split
import cv2
import serial # 导入串口包
import time

# ------------------------------定义参数------------------------------
TIME_STEPS = 1  # 时间点数据 每次读取1行共28次 same as the height of the image
INPUT_SIZE = 9  # 每行读取28个像素点 same as the width of the image
BATCH_SIZE = 128 # 每个批次训练50张图片
BATCH_INDEX = 0
OUTPUT_SIZE = 4  # 每张图片输出分类矩阵
CELL_SIZE = 128  # RNN中隐藏单元
LR = 0.001  # 学习率

# ---------------------------载入数据及预处理---------------------------
# 下载MNIST数据
# training X shape (60000, 28x28), Y shape (60000, )
# test X shape (10000, 28x28), Y shape (10000, )
def loaddata(filepath):
    all_data = np.loadtxt(filepath)
    return all_data

path = "/home/gaofei/PycharmProjects/ElectroMagnetArea/SoarFacedata/megedData.txt"
# 修改爲int8數據類型
# dataSet = loaddata(path)
dataSet = loaddata(path).astype(np.int8)
x = dataSet[:,:9]
y = dataSet[:,9:13]

# 區分訓練集合和驗證集合
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.25)
X_train = X_train.reshape((X_train.shape[0],1,X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0],1,X_test.shape[1]))
# ---------------------------创建RNN神经网络---------------------------
# 创建RNN模型
model = Sequential()

# RNN cell
model.add(SimpleRNN(
    # 设置输入batch形状 批次数量50 时间点28 每行读取像素28个
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
    batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),
    # RNN输出给后一层的结果为50
    units=CELL_SIZE,
    unroll=False
    # unroll=False
))

# output layer
model.add(Dense(OUTPUT_SIZE))  # 全连接层 输出对应10分类
model.add(Activation('softmax'))  # 激励函数 tanh

# ---------------------------神经网络优化器---------------------------
# optimizer
adam = adam_v2.Adam(LR)

# We add metrics to get more results you want to see
# 激活神经网络
model.compile(optimizer=adam,  # 加速神经网络
              loss='categorical_crossentropy',  # 损失函数
              metrics=['accuracy'])  # 计算误差或准确率

# --------------------------------训练和预测------------------------------
cost_list = []
acc_list = []
step_list = []
for step in range(80):
    # 分批截取数据 BATCH_INDEX初始值为0 BATCH_SIZE为50 取28个步长和28个INPUT_SIZE
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :, :]
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :]

    # 计算误差
    cost = model.train_on_batch(X_batch, Y_batch)

    # 累加参数
    BATCH_INDEX += BATCH_SIZE
    # 如果BATCH_INDEX累加大于总体的个数 则重新赋值0开始分批计算
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    # 每隔200步输出
    if step % 200 == 0:
        # 评价算法
        cost, accuracy = model.evaluate(
            X_test, y_test,
            batch_size=y_test.shape[0],
            verbose=False)
        # 写入列表
        cost_list.append(cost)
        acc_list.append(accuracy)
        step_list.append(step)
        print('test cost: ', cost, 'test accuracy: ', accuracy)

# 保存模型
model.save('KerasRnn.h5')
newmodel = keras.models.load_model('KerasRnn.h5')
# print(newmodel.summary())


# 切換模型爲tflite格式
# https://blog.51cto.com/u_15764210/5616515
# https://www.jb51.net/article/187247.htm
import tensorflow as tf
input_graph_name = "KerasRnn.h5"
output_graph_name = 'KerasRnn.tflite'
# converter = tf.lite.TFLiteConverter.from_keras_model_file(model_file=input_graph_name)
converter = tf.lite.TFLiteConverter.from_keras_model(model=newmodel)
converter.post_training_quantize = True
#在windows平台这个函数有问题，无法正常使用
tflite_model = converter.convert()
open(output_graph_name, "wb").write(tflite_model)
print ("generate:",output_graph_name)

# 驗證lite模型
# https://www.jb51.net/article/187247.htm
# 重新加載模型
interpreter = tf.lite.Interpreter(model_path="KerasRnn.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

# 實際運行的結果
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
#         input = dataarray.reshape((1, 1, 9))
#         result = model.predict(input)
#         pred_y = np.argmax(result, 1)
#         endtime = time.time()
#
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
#
# input = X_test[0].reshape((1,1,9))
# result = model.predict(input)
# pred_y = np.argmax(result, 1)


