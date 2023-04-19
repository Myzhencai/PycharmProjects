
# -*- coding: utf-8 -*-

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam
import keras
from sklearn.model_selection import train_test_split
import cv2
import serial # 导入串口包
import time

lefteye = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/lefteye.png")
leftface = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/leftface.png")
lefthead = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/lefthead.png")
leftjaw = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/leftjaw.png")

newmodel = keras.models.load_model('my_model.h5')

ser = serial.Serial("/dev/ttyUSB0",256000,timeout = 0.01) # 开启com3口，波特率115200，超时5
ser.flushInput() # 清空缓冲区
while True:
    currentdata = ser.readline() # 获取串口缓冲区数据
    if currentdata !=b'' :
        starttime= time.time()
        currentdata = str(currentdata, 'UTF-8')
        currentdatalist = currentdata.split('\r\n')[0]
        currentdatalist = currentdatalist.split(",")
        dataarray = np.array(currentdatalist,dtype='float32').reshape((-1,9))
        input = dataarray.reshape((1, 1, 9))
        result = newmodel.predict(input)
        pred_y = np.argmax(result, 1)
        endtime = time.time()

        print("單次運算的時間:",endtime-starttime)
        if pred_y==0:
            print("當前在額頭區域")
            cv2.imshow('image', lefthead)
            cv2.waitKey(30)
        elif pred_y==1:
            # print("current is" ,areaid)
            print("當前在下頜線區域")
            cv2.imshow('image', leftjaw)
            cv2.waitKey(30)
        elif pred_y==2:
            # print("current is" ,areaid)
            print("當前在面部區域")
            cv2.imshow('image', leftface)
            cv2.waitKey(30)
        elif pred_y==3:
            # print("current is" ,areaid)
            print("當前在眼周區域")
            cv2.imshow('image', lefteye)
            cv2.waitKey(30)

# input = X_test[0].reshape((1,1,9))
# result = model.predict(input)
# pred_y = np.argmax(result, 1)
#
# # 保存模型
# model.save('my_model.h5')
#
# print(newmodel.summary())

