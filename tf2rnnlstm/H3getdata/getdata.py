import time
import serial
import numpy as np

# 左邊
serleft = serial.Serial("/dev/ttyS2",115200,timeout = 0.01)
serleft.flushInput()
# 右邊
serringht = serial.Serial("/dev/ttyS1",115200,timeout = 0.01)
serringht.flushInput()

datasaverleft = []
datasaverright = []

while True:
    currentdataleft = serleft.readline()
    currentdataright = serringht.readline()
    if currentdataleft !=b'':
        # starttime= time.time()
        # 左邊
        currentdataleft = str(currentdataleft, 'UTF-8')
        currentdatalistleft = currentdataleft.split('\n')[0]
        currentdatalistleft = currentdatalistleft.split(",")
        dataarrayleft = np.array(currentdatalistleft, dtype='float16').reshape((-1, 9))
        datasaverleft.append(dataarrayleft[0][:9])
        np.savetxt("./sensordataleft.txt", np.array(datasaverleft).reshape((-1, 9)))
        print("dataarrayleft :", dataarrayleft)
    if  currentdataright !=b'':
        currentdataright = str(currentdataright, 'UTF-8')
        currentdatalistright = currentdataright.split('\n')[0]
        currentdatalistright = currentdatalistright.split(",")
        dataarrayright = np.array(currentdatalistright, dtype='float16').reshape((-1, 9))
        datasaverright.append(dataarrayright[0][:9])
        np.savetxt("./sensordataright.txt", np.array(datasaverright).reshape((-1, 9)))
        print("dataarrayright :",dataarrayright)
    # else:
    #     print("wrong port")
        # endtime = time.time()
    time.sleep(0.001) # 延时0.1秒，免得CPU出问题

