import socket
import time
import serial
import threading
import numpy as np
# 此處代碼放在H3上面

def leftserial():
    serleft = serial.Serial("/dev/ttyS2", 115200, timeout=0.01)
    serwriteleft = serial.Serial("/dev/ttyS0", 115200, timeout=0.01)
    serleft.flushInput()
    serwriteleft.flushInput()
    while True:
        currentdataleft = serleft.readline()
        if currentdataleft != b'':
            serwriteleft.write(currentdataleft)

def rightserial():
    serright = serial.Serial("/dev/ttyS2", 115200, timeout=0.01)
    serwriteleft = serial.Serial("/dev/ttyS0", 115200, timeout=0.01)
    serwriteleft.flushInput()
    serright.flushInput()
    while True:
        currentdataright = serright.readline()
        if currentdataright != b'':
            serwriteleft.write(currentdataright)

flag = True
# 生成socket对象
client = socket.socket()
ipaddress = input("請輸入電腦網絡的ip地址：")
# 链接要链接的ip和port（端口）
client.connect((ipaddress, 6666))
# while循环
while flag:
    commandfromPC = client.recv(1024).decode()
    if commandfromPC == 'openleft':
        print("opened left serial ")
        client.send('openedleft'.encode())

    elif commandfromPC == 'openright':
        print("open right serial ")
        client.send('openedright'.encode())

    elif commandfromPC == 'closeleft':
        print("close left serial ")
        client.send('closedleft'.encode())

    elif commandfromPC == 'closeright':
        print("close right serial ")
        client.send('closedright'.encode())
    elif commandfromPC == 'closeall':
        print("close socket")
        flag = False
    else:
        print("等待PC端指令")

# 关闭socket链接
client.close()
print('Server Closed')