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
    # 根據接受到的指令來進行對應串口數據的讀取
    # 打開ttyS2串口
    msg = input("Enter your message('q' for quit):").strip()
    msg = '請問需要打開左邊串口麼？（yes or no）'

    # 判断是否为空
    if len(msg) == 0:
        print("Message can't be empty")
        continue

    # 发送数据
    client.send(msg.encode())

    # 判断是否为'q'
    if msg != 'q':

        # 接收数据
        data = client.recv(1024)

        # 打印接收到的数据
        print(data)

    else:
        # 条件为False
        flag = False

# 关闭socket链接
client.close()
print('Server Closed')