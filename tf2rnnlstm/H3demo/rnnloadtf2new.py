import tensorflow.compat.v1 as tf
import numpy as np
import time
import serial

tf.disable_eager_execution()


sess=tf.Session()
#先加载图和参数变量
saver = tf.train.import_meta_graph('/home/rer/Matrix/H3demo/savemodel/rnnlstmtf2.meta')
saver.restore(sess, tf.train.latest_checkpoint('/home/rer/Matrix/H3demo/savemodel'))

#访问placeholders变量，并且创建feed-dict来作为placeholders的新值
graph = tf.get_default_graph()
input_x = sess.graph.get_tensor_by_name('Input:0')
output = sess.graph.get_tensor_by_name('Output:0')

#加載實時數據
ser = serial.Serial("/dev/ttyS1",115200,timeout = 0.01)
ser.flushInput()
datasaver = []

while True:
    currentdata = ser.readline()
    if currentdata !=b'':
        # starttime= time.time()
        currentdata = str(currentdata, 'UTF-8')
        currentdatalist = currentdata.split('\n')[0]
        currentdatalist = currentdatalist.split(",")
        newx = np.array(currentdatalist, dtype='float16').reshape((-1, 9))
        test_output = sess.run(output, {input_x: newx})
        pred_y = np.argmax(test_output, 1)
        # endtime = time.time()
        print("pred_y: ", pred_y)
        # print("predict used:",endtime - starttime)
    time.sleep(0.001) # 延时0.1秒，免得CPU出问题