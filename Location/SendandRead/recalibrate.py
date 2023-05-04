import time
import serial
import numpy as np


# 左邊
# serleft = serial.Serial("/dev/ttyS2",115200,timeout = 0.01)
# serleft.flushInput()
# 右邊
serringht = serial.Serial("/dev/ttyS1",115200,timeout = 0.01)
serringht.flushInput()

hex_str = 'r'
serringht.write(bytes.fromhex(hex_str))
time.sleep(1)

while True:
    # currentdataleft = serleft.readline()
    currentdataright = serringht.readline()
    # if currentdataleft !=b'':
        # starttime= time.time()
        # 左邊
        # currentdataleft = str(currentdataleft, 'UTF-8')
        # currentdatalistleft = currentdataleft.split('\n')[0]
        # currentdatalistleft = currentdatalistleft.split(",")
        # dataarrayleft = np.array(currentdatalistleft, dtype='float16').reshape((-1, 9))
        # print("dataarrayleft :", dataarrayleft)
    if  currentdataright !=b'':
        print(currentdataright)