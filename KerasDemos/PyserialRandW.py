'''
功能： 发送指定AT命令到设备 实现关机
'''

import serial
import numpy as np


def write(serSend,informations):
    serSend.write(informations.encode())
    # serSend.close()


def readandwrite(reader,sender):
    currentdata = reader.readline()
    while True:
        if currentdata !=b'' :
            currentdata = str(currentdata, 'UTF-8')
            write(sender,currentdata)


if __name__ == "__main__":
    # 請自行修改對應的串口號
    serRead = serial.Serial("/dev/ttyUSB0", 256000, timeout=0.01)
    serSend = serial.Serial("COM5", 256000, timeout=0.01)
    serRead.flushInput()  # 清空缓冲区
    serSend.flushInput()  # 清空缓冲区
    readandwrite(serRead,serSend)