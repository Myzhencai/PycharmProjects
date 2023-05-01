import tensorflow.compat.v1 as tf
import numpy as np
import time
import serial

tf.disable_eager_execution()

sess=tf.Session()
#先加载图和参数变量
saver = tf.train.import_meta_graph('/home/gaofei/PycharmProjects/tf2rnnlstm/gaofeirealtimemodel/Matrix.meta')
saver.restore(sess, tf.train.latest_checkpoint('/home/gaofei/PycharmProjects/tf2rnnlstm/gaofeirealtimemodel'))

# 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
graph = tf.get_default_graph()
input_x = sess.graph.get_tensor_by_name('Input:0')
output = sess.graph.get_tensor_by_name('Output:0')

#加載實時數據
ser = serial.Serial("/dev/ttyUSB0",256000,timeout = 0.01) # 开启com3口，波特率115200，超时5
ser.flushInput() # 清空缓冲区
while True:
    currentdata = ser.readline() # 获取串口缓冲区数据
    if currentdata !=b'' and currentdata !=b'\n':
        starttime= time.time()
        currentdata = str(currentdata, 'UTF-8')
        currentdatalist = currentdata.split('\r\n')[0]
        currentdatalist = currentdatalist.split(",")
        dataarray = np.array(currentdatalist,dtype='float32').reshape((-1,18))
        newx = dataarray[:, :18]
        test_output = sess.run(output, {input_x: newx})
        pred_y = np.argmax(test_output, 1)
        endtime = time.time()

        print("單次運算的時間:",endtime-starttime)
        if pred_y==0:
            print("當前在額頭區域")
        elif pred_y==1:
            # print("current is" ,areaid)
            print("當前在左下頜線區域")
        elif pred_y==2:
            # print("current is" ,areaid)
            print("當前在右下頜線區域")
        elif pred_y==3:
            # print("current is" ,areaid)
            print("當前在左邊臉部")
        elif pred_y==4:
            # print("current is" ,areaid)
            print("當前在右邊臉部")
        elif pred_y==5:
            # print("current is" ,areaid)
            print("當前在左邊眼部")
        elif pred_y==6:
            # print("current is" ,areaid)
            print("當前在右邊眼部")

    time.sleep(0.001) # 延时0.1秒，免得CPU出问题