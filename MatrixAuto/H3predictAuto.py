import tensorflow.compat.v1 as tf
import numpy as np
import threading
import serial
# 此代码实现在H3板子上对新训练的模型进行实际的预测，并将预测结果通过PCpredict得到显示
class RealtimePredict:
    realtimebuffer = [None, None]
    def __init__(self,PCclient):
        tf.disable_eager_execution()
        self.sessright = tf.Session()
        self.saverR = tf.train.import_meta_graph('/home/rer/Model/rightmodel/Matrixrightbi.meta')
        self.saverR.restore(self.sessright, tf.train.latest_checkpoint('/home/rer/Model/rightmodel'))

        # 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
        self.graph_right = tf.get_default_graph()
        self.input_x_right = self.sessright.graph.get_tensor_by_name('Input:0')
        self.output_right = self.sessright.graph.get_tensor_by_name('Output:0')

        # 左邊tf圖
        self.sessleft = tf.Session()
        self.saverL = tf.train.import_meta_graph('/home/rer/Model/leftmodel/Matrixleftbi.meta')
        self.saverL.restore(self.sessleft, tf.train.latest_checkpoint('/home/rer/Model/leftmodel'))

        # 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
        self.graph_left = tf.get_default_graph()
        self.input_x_left = self.sessleft.graph.get_tensor_by_name('Input:0')
        self.output_left = self.sessleft.graph.get_tensor_by_name('Output:0')
        self.client = PCclient

    def serialleft(self):
        serleft = serial.Serial("/dev/ttyS2", 115200, timeout=0.01)
        serleft.flushInput()
        while True:
            currentdataleft = serleft.readline()
            if currentdataleft != b'':
                currentdataleft = str(currentdataleft, 'UTF-8')
                currentdatasaverleftleftold = currentdataleft.split('\r\n')[0]
                currentdatasaverleftleft = currentdatasaverleftleftold.split(",")
                inversecurrentdatasaverleftleft = currentdatasaverleftleftold.split(",")
                inversecurrentdatasaverleftleft.reverse()
                currentdatasaverleftleft = currentdatasaverleftleft + inversecurrentdatasaverleftleft

                dataarrayleft = np.array(currentdatasaverleftleft, dtype='float32').reshape((-1, 18))

                self.realtimebuffer[0] = dataarrayleft
            else:
                self.realtimebuffer[0] = None

    # 獲取右邊臉部數據
    def serialrifht(self):
        serringht = serial.Serial("/dev/ttyS1", 115200, timeout=0.01)
        serringht.flushInput()
        while True:
            currentdataright = serringht.readline()
            if currentdataright != b'':
                currentdataleft = str(currentdataright, 'UTF-8')
                currentdatasaverleftleftold = currentdataleft.split('\r\n')[0]
                currentdatasaverleftleft = currentdatasaverleftleftold.split(",")
                inversecurrentdatasaverleftleft = currentdatasaverleftleftold.split(",")
                inversecurrentdatasaverleftleft.reverse()
                currentdatasaverleftleft = currentdatasaverleftleft + inversecurrentdatasaverleftleft
                dataarrayright = np.array(currentdatasaverleftleft, dtype='float32').reshape((-1, 18))

                self.realtimebuffer[1] = dataarrayright
            else:
                self.realtimebuffer[1] = None

    def chosendataold(self):
        # chosendataresult = np.zeros((1, 18))
        while True:
            if (self.realtimebuffer[0] is not None) and (self.realtimebuffer[1] is not None):
                leftsum = np.sum(self.realtimebuffer[0] * self.realtimebuffer[0])
                rightsum = np.sum(self.realtimebuffer[1] * self.realtimebuffer[1])

                if leftsum > rightsum:
                    chosendataresult = self.realtimebuffer[0]
                    # print(realtimebuffer[0].shape)
                    # chosendataresult =np.c_[leftkeys,realtimebuffer[0]]
                    test_output = self.sessleft.run(self.output_left, {self.input_x_left: chosendataresult})
                    pred_y = np.argmax(test_output, 1)
                    if pred_y == 0:
                        # predictareaid[0] = 0
                        print("左邊數據：當前在額頭區域")
                        self.client.send('0\r\n'.encode())
                    elif pred_y == 1:
                        # predictareaid[0] = 1
                        # print("current is" ,areaid)
                        print("左邊數據：當前在左下頜線區域")
                        self.client.send('1\r\n'.encode())
                    elif pred_y == 2:
                        # predictareaid[0] = 2
                        # print("current is" ,areaid)
                        print("左邊數據：當前在左臉部")
                        self.client.send('2\r\n'.encode())
                    elif pred_y == 3:
                        # predictareaid[0] = 3
                        # print("current is" ,areaid)
                        print("左邊數據：當前在左眼周")
                        self.client.send('3\r\n'.encode())
                else:
                    chosendataresult = self.realtimebuffer[1]
                    # print(realtimebuffer[1].shape)
                    # chosendataresult = np.c_[rightkeys, realtimebuffer[1]]
                    test_output = self.sessright.run(self.output_right, {self.input_x_right: chosendataresult})
                    pred_y = np.argmax(test_output, 1)

                    if pred_y == 0:
                        # predictareaid[1] =0
                        print("右邊數據：當前在額頭區域")
                        self.client.send('4\r\n'.encode())
                    elif pred_y == 1:
                        # predictareaid[1] = 1
                        print("右邊數據：當前在右下頜線區域")
                        self.client.send('5\r\n'.encode())
                    elif pred_y == 2:
                        # predictareaid[1] = 2
                        print("右邊數據：當前在右臉部")
                        self.client.send('6\r\n'.encode())
                    elif pred_y == 3:
                        # predictareaid[1] = 3
                        print("右邊數據：當前在右眼周")
                        self.client.send('7\r\n'.encode())

            elif (self.realtimebuffer[0] is None) and (self.realtimebuffer[1] is not None):
                chosendataresult = self.realtimebuffer[1]
                # print(realtimebuffer[1].shape)
                # chosendataresult = np.c_[rightkeys, realtimebuffer[1]]
                test_output = self.sessright.run(self.output_right, {self.input_x_right: chosendataresult})
                pred_y = np.argmax(test_output, 1)

                if pred_y == 0:
                    # predictareaid[1] =0
                    print("只有右邊數據：當前在額頭區域")
                    self.client.send('8\r\n'.encode())
                elif pred_y == 1:
                    # predictareaid[1] = 1
                    print("只有右邊數據：當前在右下頜線區域")
                    self.client.send('9\r\n'.encode())
                elif pred_y == 2:
                    # predictareaid[1] = 2
                    print("只有右邊數據：當前在右臉部")
                    self.client.send('10\r\n'.encode())
                elif pred_y == 3:
                    # predictareaid[1] = 3
                    print("只有右邊數據：當前在右眼周")
                    self.client.send('11\r\n'.encode())

            elif (self.realtimebuffer[0] is not None) and (self.realtimebuffer[1] is None):
                chosendataresult = self.realtimebuffer[0]
                # print(realtimebuffer[0].shape)
                # chosendataresult = np.c_[leftkeys, realtimebuffer[0]]
                test_output = self.sessleft.run(self.output_left, {self.input_x_left: chosendataresult})
                pred_y = np.argmax(test_output, 1)
                if pred_y == 0:
                    # predictareaid[0] = 0
                    print("只有左邊：當前在額頭區域")
                    self.client.send('12\r\n'.encode())
                elif pred_y == 1:
                    # predictareaid[0] = 1
                    # print("current is" ,areaid)
                    print("只有左邊：當前在左下頜線區域")
                    self.client.send('13\r\n'.encode())
                elif pred_y == 2:
                    # predictareaid[0] = 2
                    # print("current is" ,areaid)
                    print("只有左邊：當前在左臉部")
                    self.client.send('14\r\n'.encode())
                elif pred_y == 3:
                    # predictareaid[0] = 3
                    # print("current is" ,areaid)
                    print("只有左邊：當前在左眼周")
                    self.client.send('15\r\n'.encode())
            # 只有有數據才預測
            else:
                continue

    def startpredict(self):
        # 0 對應額頭 ,1 對應左下頜,2對應右下頜,3左邊臉部,4對應右邊臉部,5對應左邊眼周,6對應右邊臉周
        left_thread = threading.Thread(target=self.serialleft)
        right_thread = threading.Thread(target=self.serialrifht)
        # chosen_thread = threading.Thread(target=chosendata,args=(conn,leftpersonalkeys,rightpersonalkeys,))
        chosen_thread = threading.Thread(target=self.chosendataold)

        # 开启线程
        left_thread.start()
        right_thread.start()
        chosen_thread.start()
