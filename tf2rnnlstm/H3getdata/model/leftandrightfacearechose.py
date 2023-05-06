import tensorflow.compat.v1 as tf
import numpy as np
import threading
import time
import serial
# 可用的模型


tf.disable_eager_execution()

leftandrightsum=[1,1]
predictareaid = [None,None]

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
def chosendata():
    chosendataresult = np.zeros((1, 9))
    while True:
        if (realtimebuffer[0] is not None) and (realtimebuffer[1] is not None):
            leftsum = np.sum(realtimebuffer[0]*realtimebuffer[0])
            rightsum = np.sum(realtimebuffer[1]*realtimebuffer[1])
            # leftsum = abs(realtimebuffer[0][0][0])
            # rightsum = abs(realtimebuffer[1][0][0])
            if leftsum > rightsum:
                chosendataresult = realtimebuffer[0]
                test_output = sessleft.run(output_left, {input_x_left: chosendataresult})
                pred_y = np.argmax(test_output, 1)
                if pred_y == 0:
                    # predictareaid[0] = 0
                    print("當前在額頭區域")
                elif pred_y == 1:
                    # predictareaid[0] = 1
                    # print("current is" ,areaid)
                    print("當前在左下頜線區域")
                elif pred_y == 2:
                    # predictareaid[0] = 2
                    # print("current is" ,areaid)
                    print("當前在左臉部")
                elif pred_y == 3:
                    # predictareaid[0] = 3
                    # print("current is" ,areaid)
                    print("當前在左眼周")
            else:
                chosendataresult = realtimebuffer[1]
                test_output = sessright.run(output_right, {input_x_right: chosendataresult})
                pred_y = np.argmax(test_output, 1)

                if pred_y == 0:
                    # predictareaid[1] =0
                    print("當前在額頭區域")
                elif pred_y == 1:
                    # predictareaid[1] = 1
                    print("當前在右下頜線區域")
                elif pred_y == 2:
                    # predictareaid[1] = 2
                    print("當前在右臉部")
                elif pred_y == 3:
                    # predictareaid[1] = 3
                    print("當前在右眼周")

                # datalist.append(chosendataresult[0][:9])
                # print("chose one right")
        #         print(chosendataresult)
        elif (realtimebuffer[0] is None) and (realtimebuffer[1] is not None):
            chosendataresult = realtimebuffer[1]
            test_output = sessright.run(output_right, {input_x_right: chosendataresult})
            pred_y = np.argmax(test_output, 1)

            if pred_y == 0:
                # predictareaid[1] =0
                print("當前在額頭區域")
            elif pred_y == 1:
                # predictareaid[1] = 1
                print("當前在右下頜線區域")
            elif pred_y == 2:
                # predictareaid[1] = 2
                print("當前在右臉部")
            elif pred_y == 3:
                # predictareaid[1] = 3
                print("當前在右眼周")
            # datalist.append(chosendataresult[0][:9])
            # print("chose 2 right")
            # print(chosendataresult)
        elif (realtimebuffer[0] is not None) and (realtimebuffer[1] is None):
            chosendataresult = realtimebuffer[0]
            test_output = sessleft.run(output_left, {input_x_left: chosendataresult})
            pred_y = np.argmax(test_output, 1)
            if pred_y == 0:
                # predictareaid[0] = 0
                print("當前在額頭區域")
            elif pred_y == 1:
                # predictareaid[0] = 1
                # print("current is" ,areaid)
                print("當前在左下頜線區域")
            elif pred_y == 2:
                # predictareaid[0] = 2
                # print("current is" ,areaid)
                print("當前在左臉部")
            elif pred_y == 3:
                # predictareaid[0] = 3
                # print("current is" ,areaid)
                print("當前在左眼周")
        # 只有有數據才預測
        else:
            continue
        # 對選出的數據做預測


        # time.sleep(0.001)  # 延时0.1秒，免得CPU出问题

        # time.sleep(0.001)


if __name__ == '__main__':
    # 共享的buffer內存數據
    realtimebuffer = [None, None]
    # 右邊邊tf圖
    sessright = tf.Session()
    # 先加载图和参数变量
    saver = tf.train.import_meta_graph('/home/rer/H3rightfaceDemo/ModelFile/Matrixright.meta')
    saver.restore(sessright, tf.train.latest_checkpoint('/home/rer/H3rightfaceDemo/ModelFile'))

    # 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
    graph_right = tf.get_default_graph()
    input_x_right = sessright.graph.get_tensor_by_name('Input:0')
    output_right = sessright.graph.get_tensor_by_name('Output:0')

    # 左邊tf圖
    sessleft = tf.Session()
    # 先加载图和参数变量
    saver = tf.train.import_meta_graph('/home/rer/H3leftfaceDemo/ModelFile/Matrixleft.meta')
    saver.restore(sessleft, tf.train.latest_checkpoint('/home/rer/H3leftfaceDemo/ModelFile'))

    # 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
    graph_left = tf.get_default_graph()
    input_x_left = sessleft.graph.get_tensor_by_name('Input:0')
    output_left = sessleft.graph.get_tensor_by_name('Output:0')


    # 0 對應額頭 ,1 對應左下頜,2對應右下頜,3左邊臉部,4對應右邊臉部,5對應左邊眼周,6對應右邊臉周
    left_thread = threading.Thread(target=serialleft)
    right_thread = threading.Thread(target=serialrifht)
    chosen_thread = threading.Thread(target=chosendata)

    # 开启线程
    left_thread.start()
    right_thread.start()
    chosen_thread.start()
