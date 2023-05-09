import socket

flag = True
# 生成socket对象
server = socket.socket()
# 绑定ip和端口
server.bind(('172.16.2.55', 6666))
# 监听绑定的端口
server.listen()
# 方便识别打印一个我在等待
print("等待Matrix鏈接...")
# 这里用两个值接受，因为链接上之后使用的是客户端发来请求的这个实例
# 所以下面的传输要使用conn实例操作
conn, addr = server.accept()

# 打印链接成功
print('成功鏈接Matrix')

# 进入循环
while flag:
    # 打開左邊
    openleft = input("請問打開左邊的串口進行數據收集麼？（yes or no）:").strip()
    if openleft =='yes':
        # 發送信號打開左邊串口並監聽是否打開
        conn.send('openleft'.encode())
        while True:
            openedleft = conn.recv(1024).decode()
            if openedleft == 'openedleft':
                print("left serial opened ")
                break
            print("手機左邊的數據")

    data = conn.recv(1024).decode()

    # 判断
    # if data != 'q':
    #     # 打印收到数据
    #     print('收到：', data)
    #     # 发送我收到数据了
    #     conn.send('Send Successed!'.encode())
    # else:
    #     # 条件为False
    #     flag = False

# 关闭socket链接
server.close()