import socket
import time
import serial
import threading
import numpy as np
# 此處代碼放在H3上面

flag = True
# 生成socket对象
client = socket.socket()
clientcontrol = socket.socket()
# ipaddress = input("請輸入電腦網絡的ip地址：")
ipaddress = '172.16.2.55'
# 链接要链接的ip和port（端口）
client.connect((ipaddress, 6868))
print("connect1")
openleft = True
openright = True

def leftserial(client):
    serleft = serial.Serial("/dev/ttyS2", 115200, timeout=0.01)
    # serwriteleft = serial.Serial("/dev/ttyS0", 115200, timeout=0.01)
    serleft.flushInput()
    # serwriteleft.flushInput()
    while True:
        currentdataleft = serleft.readline()
        if currentdataleft != b'':
            client.send(currentdataleft)

def rightserial():
    serright = serial.Serial("/dev/ttyS2", 115200, timeout=0.01)
    serwriteleft = serial.Serial("/dev/ttyS0", 115200, timeout=0.01)
    serwriteleft.flushInput()
    serright.flushInput()
    while True:
        currentdataright = serright.readline()
        if currentdataright != b'':
            serwriteleft.write(currentdataright)

# def controlflow():
#     global openleft
#     global openright
#     clientcontrol = socket.socket()
#     clientcontrol.connect(("172.16.2.55", 9987))
#     print("connect2 ")
#     while True:
#         flowComand = clientcontrol.recv(1024).decode()
#         print(flowComand)
#         if flowComand == "closeleft":
#             openleft = False
#         if flowComand == "closeright":
#             openright = False
#
# flowcontrol = threading.Thread(target=controlflow,name='LoopThread')
# flowcontrol.start()
# while循环
while flag:
    commandfromPC = client.recv(1024).decode()
    if commandfromPC == 'openleft':
        print("打開左邊串口")
        serleft = serial.Serial("/dev/ttyS2", 115200, timeout=0.01)
        serleft.flushInput()
        client.send('openedleft'.encode())
        while openleft:
            currentdataleft = serleft.readline()
            if currentdataleft != b'':
                # print("left :",currentdataleft)
                client.send(currentdataleft)
                leftstatue = client.recv(1024).decode()
                if leftstatue ==  'leftdata':
                    print("在抓取左邊Sensor數據 ")
                if leftstatue ==  'finishleft':
                    print("完成左邊數據的抓取")
                    openleft = False
        print("關掉左邊的串口")
        serleft.close()

    elif commandfromPC == 'openright':
        print("打開右邊串口")
        serright = serial.Serial("/dev/ttyS1", 115200, timeout=0.01)
        serright.flushInput()
        client.send('openedright'.encode())
        while openright:
            currentdataright = serright.readline()
            if currentdataright != b'':
                # print("left :",currentdataleft)
                client.send(currentdataright)
                rightstatue = client.recv(1024).decode()
                if rightstatue == 'rightdata':
                    print("在抓取右邊Sensor數據 ")
                if rightstatue == 'finishright':
                    print("完成右邊數據的抓取")
                    openright = False
        print("關掉右邊的串口")
        serright.close()


    elif commandfromPC == 'closeall':
        print("close socket")
        flag = False
        client.shutdown(2)
        client.close()
        print('Server Closed')
    else:
        print("等待PC端指令")

# 关闭socket链接
