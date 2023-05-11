import socket
import numpy as np

datasaverleft = []
datasaverright = []
def getleftdata(areaid,savetest,Savepath,dataenoughnum):
    datasaverleft.clear()
    currentnum = 0
    while currentnum < dataenoughnum:
        # socket 讀取一行
        # 返回信息已經接到了數據
        conn.send('leftdata'.encode())
        currentdataleft = conn.makefile().readline()
        # 處理數據
        currentdatasaverleftleftold = currentdataleft.split('\n')[0]
        currentdatasaverleftleft = currentdatasaverleftleftold.split(",")
        inversecurrentdatasaverleftleft =currentdatasaverleftleftold.split(",")
        inversecurrentdatasaverleftleft.reverse()
        currentdatasaverleftleft = currentdatasaverleftleft + inversecurrentdatasaverleftleft
        dataarrayleft = np.array(currentdatasaverleftleft, dtype='float32').reshape((-1, 18))
        if areaid ==0:
            print("左邊額頭數據", currentnum)
            areaarray = np.array([[1,0,0,0]])
            currentdataAndarea = np.c_[dataarrayleft,areaarray]
            datasaverleft.append(currentdataAndarea[0])
            dataarray = np.array(datasaverleft,dtype='float32').reshape((-1,22))
            # print(dataarray)
            if savetest:
                np.savetxt(Savepath+"area{0}test.txt".format(areaid),dataarray)
            else:
                np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)
            currentnum +=1
        elif areaid ==1:
            print("對應左下頜", currentnum)
            areaarray = np.array([[0, 1, 0, 0]])
            currentdataAndarea = np.c_[dataarrayleft, areaarray]
            datasaverleft.append(currentdataAndarea[0])
            dataarray = np.array(datasaverleft, dtype='float32').reshape((-1, 22))
            if savetest:
                np.savetxt(Savepath + "area{0}test.txt".format(areaid), dataarray)
            else:
                np.savetxt(Savepath + "area{0}.txt".format(areaid), dataarray)
            currentnum += 1
        elif areaid ==2:
            print("對應左邊面部", currentnum)
            areaarray = np.array([[0, 0, 1, 0]])
            currentdataAndarea = np.c_[dataarrayleft, areaarray]
            datasaverleft.append(currentdataAndarea[0])
            dataarray = np.array(datasaverleft, dtype='float32').reshape((-1, 22))
            if savetest:
                np.savetxt(Savepath + "area{0}test.txt".format(areaid), dataarray)
            else:
                np.savetxt(Savepath + "area{0}.txt".format(areaid), dataarray)
            currentnum += 1
        elif areaid ==3:
            print("左邊眼周", currentnum)
            areaarray = np.array([[0, 0, 0, 1]])
            currentdataAndarea = np.c_[dataarrayleft, areaarray]
            datasaverleft.append(currentdataAndarea[0])
            dataarray = np.array(datasaverleft, dtype='float32').reshape((-1, 22))
            if savetest:
                np.savetxt(Savepath + "area{0}test.txt".format(areaid), dataarray)
            else:
                np.savetxt(Savepath + "area{0}.txt".format(areaid), dataarray)
            currentnum += 1

    return areaid


def getrightdata(areaid,savetest,Savepath,dataenoughnum):
    datasaverleft.clear()
    currentnum = 0
    while currentnum < dataenoughnum:
        # socket 讀取一行
        conn.send('rightdata'.encode())
        currentdataleft = conn.makefile().readline()
        # 返回信息已經接到了數據
        # 處理數據
        currentdatasaverleftleftold = currentdataleft.split('\n')[0]
        currentdatasaverleftleft = currentdatasaverleftleftold.split(",")
        inversecurrentdatasaverleftleft =currentdatasaverleftleftold.split(",")
        inversecurrentdatasaverleftleft.reverse()
        currentdatasaverleftleft = currentdatasaverleftleft + inversecurrentdatasaverleftleft
        dataarrayleft = np.array(currentdatasaverleftleft, dtype='float32').reshape((-1, 18))
        if areaid ==0:
            print("右邊額頭數據", currentnum)
            areaarray = np.array([[1,0,0,0]])
            currentdataAndarea = np.c_[dataarrayleft,areaarray]
            datasaverleft.append(currentdataAndarea[0])
            dataarray = np.array(datasaverleft,dtype='float32').reshape((-1,22))
            # print(dataarray)
            if savetest:
                np.savetxt(Savepath+"area{0}test.txt".format(areaid),dataarray)
            else:
                np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)
            currentnum +=1
        elif areaid ==1:
            print("對應右下頜", currentnum)
            areaarray = np.array([[0, 1, 0, 0]])
            currentdataAndarea = np.c_[dataarrayleft, areaarray]
            datasaverleft.append(currentdataAndarea[0])
            dataarray = np.array(datasaverleft, dtype='float32').reshape((-1, 22))
            if savetest:
                np.savetxt(Savepath + "area{0}test.txt".format(areaid), dataarray)
            else:
                np.savetxt(Savepath + "area{0}.txt".format(areaid), dataarray)
            currentnum += 1
        elif areaid ==2:
            print("對應右邊面部", currentnum)
            areaarray = np.array([[0, 0, 1, 0]])
            currentdataAndarea = np.c_[dataarrayleft, areaarray]
            datasaverleft.append(currentdataAndarea[0])
            dataarray = np.array(datasaverleft, dtype='float32').reshape((-1, 22))
            if savetest:
                np.savetxt(Savepath + "area{0}test.txt".format(areaid), dataarray)
            else:
                np.savetxt(Savepath + "area{0}.txt".format(areaid), dataarray)
            currentnum += 1
        elif areaid ==3:
            print("右邊眼周", currentnum)
            areaarray = np.array([[0, 0, 0, 1]])
            currentdataAndarea = np.c_[dataarrayleft, areaarray]
            datasaverleft.append(currentdataAndarea[0])
            dataarray = np.array(datasaverleft, dtype='float32').reshape((-1, 22))
            if savetest:
                np.savetxt(Savepath + "area{0}test.txt".format(areaid), dataarray)
            else:
                np.savetxt(Savepath + "area{0}.txt".format(areaid), dataarray)
            currentnum += 1

    return areaid




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

print('成功鏈接Matrix')
finishleft = False
finishright = False
enoughNum = 30
arealist =[0,1,2,3]

# 进入循环
while True:
    # 打開左邊
    if finishleft is False:
        openleft = input("請問打開左邊的串口進行數據收集麼？（yes or no）:").strip()
        if openleft =='yes':
            # 發送信號打開左邊串口並監聽H3是否打開左邊的串口
            conn.send('openleft'.encode())
            while True:
                openedleft = conn.recv(1024).decode()
                if openedleft == 'openedleft':
                    print("左邊的串口已經打開並在傳輸數據 ")
                    for areaid in arealist:
                        if areaid == 0:
                            while input("請將美容儀器開機放置在額頭區域（左），完成後請按Enter鍵 :") != '':
                                print("重新輸入")
                            print("左邊額頭數據")
                            finishedid = getleftdata(areaid,False, "./data/leftdata/", enoughNum)
                            # print(finishedid)
                        if areaid == 1 and finishedid == 0:
                            while input("請將美容儀器開機放置在下頜線區域（左），完成後請按Enter鍵 :") != '':
                                print("重新輸入")
                            print("對應左下頜")
                            finishedid = getleftdata(areaid,False, "./data/leftdata/", enoughNum)
                            # print(finishedid)
                        if areaid == 2 and finishedid == 1:
                            while input("請將美容儀器開機放置在臉部區域（左），完成後請按Enter鍵 :") != '':
                                print("重新輸入")
                            print("對應左邊面部")
                            finishedid = getleftdata(areaid,False, "./data/leftdata/", enoughNum)
                        if areaid == 3 and finishedid == 2:
                            while input("請將美容儀器開機放置在眼周區域（左），完成後請按Enter鍵 :") != '':
                                print("重新輸入")
                            print("左邊眼周")
                            finishedid = getleftdata(areaid,False, "./data/leftdata/", enoughNum)
                        if finishedid == 3:
                            print("完成左邊臉部的數據採集")
                    conn.send('finishleft'.encode())
                    closedleft= conn.recv(1024).decode()
                    # print(closedleft)
                    if closedleft == "closedleft":
                        finishleft = True
                        # 退出循環
                        break
        elif openleft =='no':
            print("不進行左邊sensor的採集")
        else:
            print("錯誤的輸入，請稍後重新輸入")

    if finishright is False:
        openright = input("請問打開右邊的串口進行數據收集麼？（yes or no）:").strip()
        if openright == 'yes':
            # 發送信號打開左邊串口並監聽是否打開
            conn.send('openright'.encode())
            while True:
                openedlright = conn.recv(1024).decode()
                if openedlright == 'openedright':
                    print("右邊的串口已經打開並在傳輸數據 ")
                    for areaid in arealist:
                        if areaid == 0:
                            while input("請將美容儀器開機放置在額頭區域（右），完成後請按Enter鍵 :") != '':
                                print("重新輸入")
                            print("右邊額頭數據")
                            finishedid = getrightdata(areaid,False, "./data/rightdata/", enoughNum)
                            # print(finishedid)
                        if areaid == 1 and finishedid == 0:
                            while input("請將美容儀器開機放置在下頜線區域（右），完成後請按Enter鍵 :") != '':
                                print("重新輸入")
                            print("對應右下頜")
                            finishedid = getrightdata(areaid,False, "./data/rightdata/", enoughNum)
                            # print(finishedid)
                        if areaid == 2 and finishedid == 1:
                            while input("請將美容儀器開機放置在臉部區域（右），完成後請按Enter鍵 :") != '':
                                print("重新輸入")
                            print("對應右邊面部")
                            finishedid = getrightdata(areaid,False, "./data/rightdata/", enoughNum)
                        if areaid == 3 and finishedid == 2:
                            while input("請將美容儀器開機放置在眼周區域（右），完成後請按Enter鍵 :") != '':
                                print("重新輸入")
                            print("右邊眼周")
                            finishedid = getrightdata(areaid,False, "./data/rightdata/", enoughNum)
                        if finishedid == 3:
                            print("完成右邊臉部的數據採集")
                    conn.send('finishright'.encode())
                    closedright = conn.recv(1024).decode()
                    if closedright == "closedright":
                       finishright = True
                    # 退出循環
                       break
        elif openright == 'no':
            print("不進行右邊sensor的採集")
        else:
            print("錯誤的輸入，請稍後重新輸入")

    if openright == 'no' and openleft =='no':
        conn.send('closeall'.encode())
        server.shutdown(2)
        server.close()
        break
    if openright == 'yes' and openleft =='yes':
        print("完成了左右面部數據採集關閉程序")
        conn.send('closeall'.encode())
        server.shutdown(2)
        server.close()
        break

