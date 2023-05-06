import time
import serial
import threading
import numpy as np

# 用於保存數據的列表
datasaverleft = []
datasaverright = []

# 獲取左邊臉部數據
def serialleft():
    # serleft = serial.Serial("/dev/ttyS2", 115200, timeout=0.01)
    serwriteleft = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.01)
    # serleft.flushInput()
    serwriteleft.flushInput()
    while True:
        currentdataleft = serwriteleft.readline()
        if currentdataleft != b'':
            print(currentdataleft)
            # serwriteleft.write(currentdataleft)

if __name__ =="__main__":
    serialleft()
