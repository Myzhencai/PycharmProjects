import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import numpy as np
import serial # 导入串口包
import time # 导入时间包
import cv2
import os

lefteye = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/lefteye.png")
leftface = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/leftface.png")
lefthead = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/lefthead.png")
leftjaw = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/leftjaw.png")

pb_file_path = "/home/gaofei/PycharmProjects/ElectroMagnetArea/demodata/"

# 重新加載網絡參與運算
sess = tf.Session()
with gfile.FastGFile(pb_file_path+'rnnlstmmodel.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='') # 导入计算图
#
# # 需要有一个初始化的过程
sess.run(tf.global_variables_initializer())
# 输入初始話
input_x = sess.graph.get_tensor_by_name('x:0')
# input_y = sess.graph.get_tensor_by_name('y:0')
# 輸出初始話
output = sess.graph.get_tensor_by_name('op_to_store:0')

# #加載實時數據
ser = serial.Serial("/dev/ttyUSB0",256000,timeout = 0.01)
ser.flushInput()
while True:
    currentdata = ser.readline()
    if currentdata !=b'' :
        # starttime= time.time()
        currentdata = str(currentdata, 'UTF-8')
        currentdatalist = currentdata.split('\r\n')[0]
        currentdatalist = currentdatalist.split(",")
        dataarray = np.array(currentdatalist,dtype='float32').reshape((-1,9))
        newx = dataarray[:, :9]
        test_output = sess.run(output, {input_x: newx})
        pred_y = np.argmax(test_output, 1)
        # endtime = time.time()
        # print("單次運算的時間:",endtime-starttime)
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
    time.sleep(0.001)
#