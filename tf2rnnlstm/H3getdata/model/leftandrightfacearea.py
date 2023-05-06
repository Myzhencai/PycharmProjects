import tensorflow.compat.v1 as tf
import numpy as np
import threading
import time
import serial

tf.disable_eager_execution()

leftandrightsum=[1,1]
predictareaid = [None,None]

def rightfacepredict():
    # global leftandrightsum
    # global predictareaid
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
            # sumvalue= np.sum(np.abs(dataarrayright[0]))
            # print("rightdatasum",sumvalue)
            # leftandrightsum[0] =np.sum(np.abs(dataarrayright[0]))
            # starttime = time.time()
            test_output = sess.run(output, {input_x: dataarrayright})
            pred_y = np.argmax(test_output, 1)
            # predictareaid[0]=pred_y
            # print("left",test_output)
            # predictconfidence[1] = test_output[pred_y]
            # endtime = time.time()
            # print("used time :",endtime-starttime)
            # print("pred_y: ",pred_y)
            if pred_y==0:
                # predictareaid[1] =0
                print("當前在額頭區域")
            elif pred_y==1:
                # predictareaid[1] = 1
                print("當前在右下頜線區域")
            elif pred_y==2:
                # predictareaid[1] = 2
                print("當前在右臉部")
            elif pred_y==3:
                # predictareaid[1] = 3
                print("當前在右眼周")
        # else:
        #     leftandrightsum[0] = 1
        #     predictareaid[0] =None


def leftfacepredict():
    # global leftandrightsum
    # global predictareaid
    sess=tf.Session()
    #先加载图和参数变量
    saver = tf.train.import_meta_graph('/home/rer/H3leftfaceDemo/ModelFile/Matrixleft.meta')
    saver.restore(sess, tf.train.latest_checkpoint('/home/rer/H3leftfaceDemo/ModelFile'))

    # 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
    graph = tf.get_default_graph()
    input_x = sess.graph.get_tensor_by_name('Input:0')
    output = sess.graph.get_tensor_by_name('Output:0')
    # #加載實時數據
    serringht = serial.Serial("/dev/ttyS2", 115200, timeout=0.01)
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
            # print("dataarrayleft :", dataarrayright)
            # sumvalue = np.sum(np.abs(dataarrayright[0]))
            # print("leftdatasum", sumvalue)
            # leftandrightsum[1] = np.sum(np.abs(dataarrayright[0]))
            # starttime = time.time()
            test_output = sess.run(output, {input_x: dataarrayright})
            pred_y = np.argmax(test_output, 1)
            # predictareaid[1] = pred_y
            # print("right",test_output)
            # predictconfidence[0] = test_output[pred_y]
            # endtime = time.time()
            # print("used time :",endtime-starttime)
            # print("pred_y: ",pred_y)
            if pred_y==0:
                # predictareaid[0] = 0
                print("當前在額頭區域")
            elif pred_y==1:
                # predictareaid[0] = 1
                # print("current is" ,areaid)
                print("當前在左下頜線區域")
            elif pred_y==2:
                # predictareaid[0] = 2
                # print("current is" ,areaid)
                print("當前在左臉部")
            elif pred_y==3:
                # predictareaid[0] = 3
                # print("current is" ,areaid)
                print("當前在左眼周")
        # else:
        #     leftandrightsum[1] = 1
        #     predictareaid[1] = None

def chosearea():
    global leftandrightsum
    global predictareaid
    while True:
        # print("right ",leftandrightsum[0])
        # print("left ",leftandrightsum[1])
        if leftandrightsum[0]/leftandrightsum[1]>1.:
            print("chose ritht")
            if predictareaid[0]==0:
                # predictareaid[1] =0
                print("當前在額頭區域")
            elif predictareaid[0]==1:
                # predictareaid[1] = 1
                print("當前在右下頜線區域")
            elif predictareaid[0]==2:
                # predictareaid[1] = 2
                print("當前在右臉部")
            elif predictareaid[0]==3:
                # predictareaid[1] = 3
                print("當前在右眼周")
        elif leftandrightsum[1]/leftandrightsum[0]>1.:
            print("chose left")
            if predictareaid[1]==0:
                # predictareaid[0] = 0
                print("當前在額頭區域")
            elif predictareaid[1]==1:
                # predictareaid[0] = 1
                # print("current is" ,areaid)
                print("當前在左下頜線區域")
            elif predictareaid[1]==2:
                # predictareaid[0] = 2
                # print("current is" ,areaid)
                print("當前在左臉部")
            elif predictareaid[1]==3:
                # predictareaid[0] = 3
                # print("current is" ,areaid)
                print("當前在左眼周")

    # print("hello")
    # print("left confidence ",predictconfidence[1])
    # print("right confidence ",predictconfidence[0])
    # if predictconfidence[0] > predictconfidence[1]:
    #     if predictareaid[0] is not None:
    #         if predictareaid[0]==0:
    #             print("當前在額頭區域")
    #         elif predictareaid[0]==1:
    #             print("當前在左下頜線區域")
    #         elif predictareaid[0]==2:
    #             print("當前在左臉部")
    #         elif predictareaid[0] ==3:
    #             print("當前在左眼周")




if __name__=="__main__":
    left_thread = threading.Thread(target=leftfacepredict)
    right_thread = threading.Thread(target=rightfacepredict)
    # chose_thread = threading.Thread(target=chosearea)

    # 开启线程
    left_thread.start()
    right_thread.start()
    # chose_thread.start()