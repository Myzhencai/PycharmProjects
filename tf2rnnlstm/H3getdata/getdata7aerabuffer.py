import time
import serial
import threading
import numpy as np

#共享的buffer內存數據
realtimebuffer = [None,None]
# 用於保存數據的列表
datasaver =[]

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
            # leftsum = np.sum(realtimebuffer[0]*realtimebuffer[0])
            # rightsum = np.sum(realtimebuffer[1]*realtimebuffer[1])
            leftsum =abs(realtimebuffer[0][0][0])
            rightsum = abs(realtimebuffer[1][0][0])
            if leftsum > rightsum:
                chosendataresult = realtimebuffer[0]
                datasaver.append(chosendataresult[0][:9])
                print("chose one left")
        #         print(chosendataresult)
            else:
                chosendataresult = realtimebuffer[1]
                datasaver.append(chosendataresult[0][:9])
                print("chose one right")
        #         print(chosendataresult)
        elif (realtimebuffer[0] is None) and (realtimebuffer[1] is not None):
            chosendataresult = realtimebuffer[1]
            datasaver.append(chosendataresult[0][:9])
            print("chose 2 right")
            # print(chosendataresult)
        elif (realtimebuffer[0] is not None) and (realtimebuffer[1] is None):
            chosendataresult = realtimebuffer[0]
            datasaver.append(chosendataresult[0][:9])
            print("chose 2 left")
            # print(chosendataresult)
        # 只有有數據才保存
        else:
            continue
        np.savetxt("./sensordata.txt", np.array(datasaver).reshape((-1, 9)))
        # print(chosendataresult)
        # time.sleep(0.033)
        # time.sleep(0.033)



if __name__ == '__main__':
    left_thread = threading.Thread(target=serialleft)
    right_thread = threading.Thread(target=serialrifht)
    chosen_thread = threading.Thread(target=chosendata)

    # 开启线程
    left_thread.start()
    right_thread.start()
    chosen_thread.start()
