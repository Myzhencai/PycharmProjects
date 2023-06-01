import socket
import psutil
from H3predictAuto import *

flag = True

# 獲取自身wifi的ip地址
info = psutil.net_if_addrs()
# 網卡名稱可以配置爲一個定值得
wlan=info['wlx90de802a9e7d']
# wlan=info['wlxc46e7b0533aa']
ipaddresslist = str(wlan[0]).split(",")
ipaddress = ipaddresslist[1].split('=')[1]
ipaddress = ipaddress[1:-1]
print("當前的wifi地址爲：",ipaddress)


server = socket.socket()
# 绑定ip和端口
server.bind((ipaddress, 6868))
# 监听绑定的端口
server.listen()
# 方便识别打印一个我在等待給一些指示燈啥的
print("等待PC斷鏈接鏈接...")
conn, addr = server.accept()
print('成功鏈接PC端')


openleft = True
openright = True



while flag:
    commandfromPC = conn.recv(1024).decode()
    if commandfromPC == 'openleft':
        print("打開左邊串口")
        serleft = serial.Serial("/dev/ttyS2", 115200, timeout=0.01)
        serleft.flushInput()
        conn.send('openedleft'.encode())
        while openleft:
            currentdataleft = serleft.readline()
            # print("")
            if currentdataleft != b'':
                # print("left :",currentdataleft)
                leftstatue = conn.recv(1024).decode()
                if leftstatue ==  'leftdata':
                    conn.send(currentdataleft)
                    print("在抓取左邊Sensor數據 ")
                if leftstatue ==  'finishleft':
                    print("完成左邊數據的抓取")
                    openleft = False
        print("關掉左邊的串口")
        serleft.close()
        conn.send('closedleft'.encode())

    elif commandfromPC == 'openright':
        print("打開右邊串口")
        serright = serial.Serial("/dev/ttyS1", 115200, timeout=0.01)
        serright.flushInput()
        conn.send('openedright'.encode())
        while openright:
            currentdataright = serright.readline()
            if currentdataright != b'':
                # print("left :",currentdataleft)
                rightstatue = conn.recv(1024).decode()
                if rightstatue == 'rightdata':
                    conn.send(currentdataright)
                    print("在抓取右邊Sensor數據 ")
                if rightstatue == 'finishright':
                    print("完成右邊數據的抓取")
                    conn.send('closedright'.encode())
                    openright = False
        print("關掉右邊的串口")
        serright.close()
    elif commandfromPC == 'closeall':
        print("close socket")
        flag = False
        server.shutdown(2)
        server.close()
        print('Server Closed')
    elif commandfromPC =='startpredict':
        print("start predict")
        print("加載新的模型")
        RealtimePredictor = RealtimePredict(conn)
        print("完成模型加載")
        conn.send('predictstarted'.encode())
        RealtimePredictor.startpredict()
    else:
        print("等待PC端指令")

# 关闭socket链接
