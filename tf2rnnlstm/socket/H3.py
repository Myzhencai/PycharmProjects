import socket
import time
import serial
import threading
import numpy as np
# 此處代碼放在H3上面

flag = True

# 生成socket对象
server = socket.socket()

# 绑定ip和端口
server.bind(('172.16.2.55', 6868))

# 监听绑定的端口
server.listen()

# 方便识别打印一个我在等待
print("等待Matrix鏈接...")

conn, addr = server.accept()

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
                leftstatue = client.recv(1024).decode()
                if leftstatue ==  'leftdata':
                    client.send(currentdataleft)
                    print("在抓取左邊Sensor數據 ")
                if leftstatue ==  'finishleft':
                    print("完成左邊數據的抓取")
                    openleft = False
        print("關掉左邊的串口")
        serleft.close()
        client.send('closedleft'.encode())

    elif commandfromPC == 'openright':
        print("打開右邊串口")
        serright = serial.Serial("/dev/ttyS1", 115200, timeout=0.01)
        serright.flushInput()
        client.send('openedright'.encode())
        while openright:
            currentdataright = serright.readline()
            if currentdataright != b'':
                # print("left :",currentdataleft)
                rightstatue = client.recv(1024).decode()
                if rightstatue == 'rightdata':
                    client.send(currentdataright)
                    print("在抓取右邊Sensor數據 ")
                if rightstatue == 'finishright':
                    print("完成右邊數據的抓取")
                    client.send('closedright'.encode())
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
