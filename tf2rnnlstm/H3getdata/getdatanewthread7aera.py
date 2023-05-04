import threading
import time
import serial
import numpy as np

# 共享數據可能需要限制大小
currentdata =[]

leftimestamp = []
rithttimestamp = []

leftdataque = []
rightdataque = []

# 左邊串口
def serialleft():
    serleft = serial.Serial("/dev/ttyS2", 115200, timeout=0.01)
    serleft.flushInput()
    while True:
        currentdataleft = serleft.readline()
        if currentdataleft != b'':
            currentdataleft = str(currentdataleft, 'UTF-8')
            currentdatalistleft = currentdataleft.split('\r\n')[0]
            currentdatalistleft = currentdatalistleft.split(",")
            dataarrayleft = np.array(currentdatalistleft, dtype='float16').reshape((-1, 9))
            currentime = time.time()
            leftimestamp.append(currentime)
            sumleft = np.sum(dataarrayleft)
            # print("sumleft :", sumleft)
            leftdataque.append(dataarrayleft)

# 右邊串口
def serialrifht():
    serringht = serial.Serial("/dev/ttyS1", 115200, timeout=0.01)
    serringht.flushInput()
    while True:
        currentdataright = serringht.readline()
        if currentdataright != b'':
            currentdataright = str(currentdataright, 'UTF-8')
            currentdatalistright = currentdataright.split('\r\n')[0]
            currentdatalistright = currentdatalistright.split(",")
            dataarrayright = np.array(currentdatalistright, dtype='float16').reshape((-1, 9))
            currentime = time.time()
            rithttimestamp.append(currentime)
            sumright = np.sum(dataarrayright)
            # print("sumright :", sumright)
            rightdataque.append(dataarrayright)

def chosendata():
    while True:
        # 先判定數據存在
        if len(leftimestamp) >0 and len(rithttimestamp)>0:
            lefttime = leftimestamp.pop(0)
            righttime = rithttimestamp.pop(0)
            dataleft = leftdataque.pop(0)
            dataright = rightdataque.pop(0)
            dataleftsum = np.sum(dataleft)
            datarightsum = np.sum(dataright)
            print("different time:",abs(lefttime-righttime))

            if abs(lefttime-righttime) <20:
                if dataleftsum > datarightsum:
                    currentdata = dataleft
                    print("currentdata chosen : left", )
                else:
                    currentdata = dataright
                    print("currentdata chosen : right", )
            else:
                print("衝過20毫秒不同步處理扔掉")

        # print(rithttimestamp)

if __name__ == '__main__':
    left_thread = threading.Thread(target=serialleft)
    right_thread = threading.Thread(target=serialrifht)
    chosen_thread = threading.Thread(target=chosendata)

    # 开启线程
    left_thread.start()
    right_thread.start()
    chosen_thread.start()
