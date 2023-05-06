import time
import serial
import threading
import numpy as np
import tensorflow.compat.v1 as tf

#共享的buffer內存數據
realtimebuffer = [None,None]
# 用於保存數據的列表
datalist =[]

# 創建tensorflow 圖
tf.disable_eager_execution()
sess=tf.Session()

#先加载图和参数变量
saver = tf.train.import_meta_graph('/home/rer/H3Demo/Modelfile/Matrixnew.meta')
saver.restore(sess, tf.train.latest_checkpoint('/home/rer/H3Demo/Modelfile'))

# 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
graph = tf.get_default_graph()
input_x = sess.graph.get_tensor_by_name('Input:0')
output = sess.graph.get_tensor_by_name('Output:0')


# 獲取左邊臉部數據
def serialleft():
    serleft = serial.Serial("/dev/ttyS2", 115200, timeout=0.01)
    serleft.flushInput()
    while True:
        currentdataleft = serleft.readline()
        if currentdataleft != b'':
            currentdataleft = str(currentdataleft, 'UTF-8')
            currentdatalistleft = currentdataleft.split('\r\n')[0]
            currentdatalistleft = currentdatalistleft.split(",")
            dataarrayleft = np.array(currentdatalistleft, dtype='float32').reshape((-1, 9))
            # 可能要枷鎖
            realtimebuffer[0] = dataarrayleft
            # print(dataarrayleft)
        else:
            realtimebuffer[0] = None

# 獲取右邊臉部數據
def serialrifht():
    serringht = serial.Serial("/dev/ttyS1", 115200, timeout=0.01)
    serringht.flushInput()
    while True:
        currentdataright = serringht.readline()
        if currentdataright != b'':
            currentdataright = str(currentdataright, 'UTF-8')
            currentdatalistright = currentdataright.split('\r\n')[0]
            currentdatalistright = currentdatalistright.split(",")
            dataarrayright = np.array(currentdatalistright, dtype='float32').reshape((-1, 9))
            realtimebuffer[1] = dataarrayright
        else:
            realtimebuffer[1] = None

# 從共享內存裏提取數據
def chosendata(areaid,savetest,Savepath):
    chosendataresult = np.zeros((1, 9))
    while True:
        if (realtimebuffer[0] is not None) and (realtimebuffer[1] is not None):
            # leftsum = np.sum(realtimebuffer[0]*realtimebuffer[0])
            # rightsum = np.sum(realtimebuffer[1]*realtimebuffer[1])
            leftsum =abs(realtimebuffer[0][0][0])
            rightsum = abs(realtimebuffer[1][0][0])
            if leftsum > rightsum:
                chosendataresult = realtimebuffer[0]
                # datalist.append(chosendataresult[0][:9])
                # print("chose one left")
        #         print(chosendataresult)
            else:
                chosendataresult = realtimebuffer[1]
                # datalist.append(chosendataresult[0][:9])
                # print("chose one right")
        #         print(chosendataresult)
        elif (realtimebuffer[0] is None) and (realtimebuffer[1] is not None):
            chosendataresult = realtimebuffer[1]
            # datalist.append(chosendataresult[0][:9])
            # print("chose 2 right")
            # print(chosendataresult)
        elif (realtimebuffer[0] is not None) and (realtimebuffer[1] is None):
            chosendataresult = realtimebuffer[0]
            # datalist.append(chosendataresult[0][:9])
            # print("chose 2 left")
            # print(chosendataresult)
        # 只有有數據才預測
        else:
            continue
        # 對選出的數據做預測
        if chosendataresult is not None:
            starttime = time.time()
            test_output = sess.run(output, {input_x: chosendataresult})
            pred_y = np.argmax(test_output, 1)
            endtime = time.time()
            print("單次運算的時間:", endtime - starttime)
            if pred_y == 0:
                print("當前在額頭區域")
            elif pred_y == 1:
                # print("current is" ,areaid)
                print("當前在左下頜線區域")
            elif pred_y == 2:
                # print("current is" ,areaid)
                print("當前在右下頜線區域")
            elif pred_y == 3:
                # print("current is" ,areaid)
                print("當前在左邊臉部")
            elif pred_y == 4:
                # print("current is" ,areaid)
                print("當前在右邊臉部")
            elif pred_y == 5:
                # print("current is" ,areaid)
                print("當前在左邊眼部")
            elif pred_y == 6:
                # print("current is" ,areaid)
                print("當前在右邊眼部")

        time.sleep(0.001)  # 延时0.1秒，免得CPU出问题
            
        # time.sleep(0.001)




if __name__ == '__main__':
    # 0 對應額頭 ,1 對應左下頜,2對應右下頜,3左邊臉部,4對應右邊臉部,5對應左邊眼周,6對應右邊臉周
    area = 0
    # test = True
    test = False
    savepath = "./7Areadata/"

    left_thread = threading.Thread(target=serialleft)
    right_thread = threading.Thread(target=serialrifht)
    chosen_thread = threading.Thread(target=chosendata,args=(area,test,savepath,))

    # 开启线程
    left_thread.start()
    right_thread.start()
    chosen_thread.start()
