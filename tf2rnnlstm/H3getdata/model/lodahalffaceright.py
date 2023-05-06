import tensorflow.compat.v1 as tf
import numpy as np
import time
import serial

tf.disable_eager_execution()


sess=tf.Session()
#先加载图和参数变量
saver = tf.train.import_meta_graph('/home/rer/H3rightfaceDemo/ModelFile/Matrixright.meta')
saver.restore(sess, tf.train.latest_checkpoint('/home/rer/H3rightfaceDemo/ModelFile'))

# 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
graph = tf.get_default_graph()
input_x = sess.graph.get_tensor_by_name('Input:0')
output = sess.graph.get_tensor_by_name('Output:0')
# #加載實時數據
serringht = serial.Serial("/dev/ttyS1", 115200, timeout=0.01)
serringht.flushInput()
while True:
    currentdataright = serringht.readline()
    if currentdataright != b'':
        currentdataright = str(currentdataright, 'UTF-8')
        currentdatasaverleftright = currentdataright.split('\r\n')[0]
        currentdatasaverleftright = currentdatasaverleftright.split(",")
        dataarrayright = np.array(currentdatasaverleftright, dtype='float32').reshape((-1, 9))
        # datasaverright.append(dataarrayright[0][:9])
        # np.savetxt("./sensordataright.txt", np.array(datasaverright).reshape((-1, 9)))
        # print("dataarrayright :", dataarrayright)
        starttime = time.time()
        test_output = sess.run(output, {input_x: dataarrayright})
        pred_y = np.argmax(test_output, 1)
        endtime = time.time()
        print("used time :",endtime-starttime)
        print("pred_y: ",pred_y)
        if pred_y==0:
            print("當前在額頭區域")
        elif pred_y==1:
            # print("current is" ,areaid)
            print("當前在右下頜線區域")
        elif pred_y==2:
            # print("current is" ,areaid)
            print("當前在右臉部")
        elif pred_y==3:
            # print("current is" ,areaid)
            print("當前在右眼周")