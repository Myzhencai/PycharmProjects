import tensorflow as tf
import numpy as np
import serial # 导入串口包
import time # 导入时间包
import cv2


# 驗證通過可以保存成模型參數形式

sess=tf.Session()
#先加载图和参数变量
saver = tf.train.import_meta_graph('/home/gaofei/PycharmProjects/ElectroMagnetArea/demodata/ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('/home/gaofei/PycharmProjects/ElectroMagnetArea/demodata'))

# lefteye = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/lefteye.png")
# leftface = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/leftface.png")
# lefthead = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/lefthead.png")
# leftjaw = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/leftjaw.png")


# 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
graph = tf.get_default_graph()
input_x = sess.graph.get_tensor_by_name('x:0')
output = sess.graph.get_tensor_by_name('op_to_store:0')
# #加載實時數據
# ser = serial.Serial("/dev/ttyUSB0",256000,timeout = 0.01)
# ser.flushInput()

while True:
    dataarray = np.array([234,245,563,226,632,2674,2456,246,2345]).reshape((-1,9))
    newx = dataarray[:, :9]
    test_output = sess.run(output, {input_x: newx})
    pred_y = np.argmax(test_output, 1)
    time.sleep(0.001)



# while True:
#     currentdata = ser.readline()
#     if currentdata !=b'' :
#         # starttime= time.time()
#         currentdata = str(currentdata, 'UTF-8')
#         currentdatalist = currentdata.split('\r\n')[0]
#         currentdatalist = currentdatalist.split(",")
#         dataarray = np.array(currentdatalist,dtype='float32').reshape((-1,9))
#         newx = dataarray[:, :9]
#         test_output = sess.run(output, {input_x: newx})
#         pred_y = np.argmax(test_output, 1)
#         # endtime = time.time()
#         # print("單次運算的時間:",endtime-starttime)
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
#     time.sleep(0.001)
