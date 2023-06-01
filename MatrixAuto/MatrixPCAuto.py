from tkinter import *
from tkinter import messagebox
import os
import socket
from TrainModel import *
from paramiko import SSHClient
from scp import SCPClient
import cv2
import time

# 本代码通过交互界面设置对应的数据抓取选项来实现数据自动化抓取（后续有时间改为单独的button控制单独的运算）
Matrixip = None
getleftdatas = False
getrightdatas = False
trainLeft = False
trainRight = False

# 左右臉部數據抓取部分
datasaverleft = []
datasaverright = []



def getleftdata(areaid,savetest,Savepath,dataenoughnum,client):
    datasaverleft.clear()
    currentnum = 0
    while currentnum < dataenoughnum:
        # socket 讀取一行
        # 返回信息已經接到了數據
        client.send('leftdata'.encode())
        currentdataleft = client.makefile().readline()
        # 處理數據
        currentdatasaverleftleftold = currentdataleft.split('\r\n')[0]
        currentdatasaverleftleft = currentdatasaverleftleftold.split(",")
        inversecurrentdatasaverleftleft =currentdatasaverleftleftold.split(",")
        inversecurrentdatasaverleftleft.reverse()
        currentdatasaverleftleft = currentdatasaverleftleft + inversecurrentdatasaverleftleft
        # print("currentdatasaverleftleft", currentdatasaverleftleft)
        if len(currentdatasaverleftleft)>18:
            continue
        else:
            dataarrayleft = np.array(currentdatasaverleftleft, dtype='float32').reshape((-1, 18))
        # print("dataarrayleft", dataarrayleft)
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


def getrightdata(areaid,savetest,Savepath,dataenoughnum,client):
    datasaverleft.clear()
    currentnum = 0
    while currentnum < dataenoughnum:
        # socket 讀取一行
        client.send('rightdata'.encode())
        currentdataleft = client.makefile().readline()
        # 返回信息已經接到了數據

        # 處理數據
        currentdatasaverleftleftold = currentdataleft.split('\n')[0]
        currentdatasaverleftleft = currentdatasaverleftleftold.split(",")
        inversecurrentdatasaverleftleft =currentdatasaverleftleftold.split(",")
        inversecurrentdatasaverleftleft.reverse()
        currentdatasaverleftleft = currentdatasaverleftleft + inversecurrentdatasaverleftleft
        # print("currentdatasaverleftleft", currentdatasaverleftleft)
        dataarrayleft = np.array(currentdatasaverleftleft, dtype='float32').reshape((-1, 18))
        # print("dataarrayright",dataarrayleft)
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

# 获取matrix ip地址（手动填写后续希望自动填写）
def connectMatrixCallBack():
    global Matrixip

    myWindow = Tk()
    #设置标题
    myWindow.title('配置链接Matrix')#标签控件布局
    Label(myWindow, text="Matrix的IP:").grid(row=0)
    entry1=Entry(myWindow)
    entry1.grid(row=0, column=1)

    # 提示用户输入IP地址并链接
    def setMatrixip():
        global Matrixip
        # 拿到ip后链接网络
        Matrixips = entry1.get()
        Matrixip = Matrixips
        # 提示用户ip地址（后续添加判断）
        msg = messagebox.askokcancel("Matrix ip",Matrixip)
        if msg:
            print("ip is",Matrixip)
            deletewindow()
        else:
            deletewindow()
            print("do not get ip")
    # 关闭窗口
    def deletewindow():
        myWindow.destroy()

    Button(myWindow, text='退出配置', command=deletewindow).grid(row=2, column=0, sticky=W, padx=5,pady=5)
    Button(myWindow, text='确认IP', command=setMatrixip).grid(row=2, column=1, sticky=W, padx=5, pady=5)
    myWindow.mainloop()


def StartFlow():
    global Matrixip
    global getleftdatas
    global getrightdatas
    global trainLeft
    global trainRight

    # 数据自动抓取部分
    flag = True
    # 生成socket对象
    client = socket.socket()
    # ipaddress = input("請輸入電腦網絡的ip地址（更具命令行得到的結果）：")
    # ipaddress = '172.16.2.52'
    if Matrixip is None:
        messagebox.showinfo("請先設置Matrix的IP地址",
                            "請先設置Matrix的IP地址，然後點擊開始整體流程")
    ipaddress = Matrixip
    # 先写定然后再操作
    # ipaddress = '192.168.43.23'
    # ipaddress = '192.168.43.168'
    # 链接要链接的ip和port（端口）
    print(Matrixip)
    if Matrixip is not None:
        client.connect((ipaddress, 6868))
        # print("鏈接上了Matrix")
        messagebox.showinfo("链接上Matrix", "已经链接上了PC与Matrix")
    else:
        return
    if getleftdatas:
        finishleft = False
    else:
        finishleft = True

    if getrightdatas:
        finishright = False
    else:
        finishright = True

    enoughNum = 60
    arealist = [0, 1, 2, 3]

    # 清除路徑下的原始數據
    savePathlist = ["./Data/leftdata/", "./Data/rightdata/"]


    if finishleft:
        pass
    else:
        dir = savePathlist[0]
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))

    if finishright:
        pass
    else:
        dir = savePathlist[1]
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))

    # 进入循环
    while True:
        # 打開左邊
        if finishleft is False:
            msgleft = messagebox.askokcancel("左边脸部数据采集确认",'我们将采集左脸数据，请确认')
            if msgleft:
                # 發送信號打開左邊串口並監聽H3是否打開左邊的串口
                client.send('openleft'.encode())
                while True:
                    openedleft = client.recv(1024).decode()
                    if openedleft == 'openedleft':
                        messagebox.showinfo("左边面部数据采集","左邊的串口已經打開並在傳輸數據")
                        # print("左邊的串口已經打開並在傳輸數據 ")
                        for areaid in arealist:
                            if areaid == 0:
                                # while input("請將美容儀器開機放置在額頭區域（左），完成後請按Enter鍵 :") != '':
                                #     print("重新輸入")
                                messagebox.showinfo("左边额头数据",
                                                    "請將美容儀器開機放置在額頭區域（左），完成後請确定按鍵")
                                finishedid = getleftdata(areaid, False, "./Data/leftdata/", enoughNum, client)
                                # print(finishedid)
                            if areaid == 1 and finishedid == 0:
                                messagebox.showinfo("左边下颌数据",
                                                    "請將美容儀器開機放置在下颌线區域（左），完成後請确定按鍵")
                                finishedid = getleftdata(areaid, False, "./Data/leftdata/", enoughNum, client)
                                # print(finishedid)
                            if areaid == 2 and finishedid == 1:
                                messagebox.showinfo("左边脸部数据",
                                                    "請將美容儀器開機放置在脸部區域（左），完成後請确定按鍵")
                                finishedid = getleftdata(areaid, False, "./Data/leftdata/", enoughNum, client)
                            if areaid == 3 and finishedid == 2:
                                messagebox.showinfo("左边眼周数据",
                                                    "請將美容儀器開機放置在眼周區域（左），完成後請确定按鍵")
                                finishedid = getleftdata(areaid, False, "./Data/leftdata/", enoughNum, client)
                            if finishedid == 3:
                                messagebox.showinfo("完成左脸",
                                                    "完成左邊臉部的數據採集，請确定按鍵")
                        client.send('finishleft'.encode())
                        closedleft = client.recv(1024).decode()
                        if closedleft == "closedleft":
                            finishleft = True
                            # 退出循環
                            break
            else:
                messagebox.showinfo("不采集左边脸部数据")

        if finishright is False:
            msgright = messagebox.askokcancel("右边脸部数据采集确认", '我们将采集右脸数据，请确认')
            if msgright:
                # 發送信號打開左邊串口並監聽是否打開
                client.send('openright'.encode())
                while True:
                    openedlright = client.recv(1024).decode()
                    if openedlright == 'openedright':
                        messagebox.showinfo("右边面部数据采集", "右邊的串口已經打開並在傳輸數據")
                        for areaid in arealist:
                            if areaid == 0:
                                messagebox.showinfo("右邊額頭數據",
                                                    "請將美容儀器開機放置在額頭區域（右），完成後請确定按鍵")
                                finishedid = getrightdata(areaid, False, "./Data/rightdata/", enoughNum,client)
                                # print(finishedid)
                            if areaid == 1 and finishedid == 0:
                                messagebox.showinfo("右边下颌数据",
                                                    "請將美容儀器開機放置在下颌线區域（右），完成後請确定按鍵")
                                finishedid = getrightdata(areaid, False, "./Data/rightdata/", enoughNum,client)
                                # print(finishedid)
                            if areaid == 2 and finishedid == 1:
                                messagebox.showinfo("右边脸部数据",
                                                    "請將美容儀器開機放置在脸部區域（右），完成後請确定按鍵")
                                finishedid = getrightdata(areaid, False, "./Data/rightdata/", enoughNum,client)
                            if areaid == 3 and finishedid == 2:
                                messagebox.showinfo("右边眼周数据",
                                                    "請將美容儀器開機放置在眼周區域（右），完成後請确定按鍵")
                                finishedid = getrightdata(areaid, False, "./Data/rightdata/", enoughNum,client)
                            if finishedid == 3:
                                messagebox.showinfo("完成右脸",
                                                    "完成右邊臉部的數據採集，請确定按鍵")
                        client.send('finishright'.encode())
                        closedright = client.recv(1024).decode()
                        if closedright == "closedright":
                            finishright = True
                            # 退出循環
                            break
            else:
                messagebox.showinfo("不采集右边脸部数据")

        if msgright is False and msgleft is False:
            messagebox.showinfo("退出数据抓取",
                                "不記錄任何數據")
            # client.send('closeall'.encode())
            # client.shutdown(2)
            # client.close()
            break
        if msgright is True and msgleft is True:
            messagebox.showinfo("退出数据抓取",
                                "完成了左右面部數據採集關閉程序")
            # client.send('closeall'.encode())
            # client.shutdown(2)
            # client.close()
            break

    # 开始训练网络
    trainleftmodelflag = messagebox.askokcancel("左脸模型", '训练左边脸部模型，请确认')
    if trainleftmodelflag:
        filePath = "./Data/leftdata/"
        savepath = "./Model/leftmodel/Matrixleftbi"
        AutoTrainLeftface = TrainModel(filePath, savepath)
        AutoTrainLeftface.trainmodel()
        messagebox.showinfo("左脸模型训练",
                            "完成左边脸部模型的训练")
    else:
        messagebox.showinfo("左脸模型训练",
                            "左边脸部模型没有训练需要手动训练")

    trainrightmodelflag = messagebox.askokcancel("右脸模型", '训练右边脸部模型，请确认')
    if trainrightmodelflag:
        filePath = "./Data/rightdata/"
        savepath = "./Model/rightmodel/Matrixrightbi"
        AutoTrainRightface = TrainModel(filePath, savepath)
        AutoTrainRightface.trainmodel()
        messagebox.showinfo("右脸模型训练",
                            "完成右边脸部模型的训练")
    else:
        messagebox.showinfo("右脸模型训练",
                            "右边脸部模型没有训练需要手动训练")

    messagebox.showinfo("替换模型的Checkpoint路径",
                        "替换模型的Checkpoint路径")

    with open("./Model/leftmodel/checkpoint",'w') as f:
        f.write('model_checkpoint_path: "/home/rer/Model/leftmodel/Matrixleftbi"\n')
        f.write('all_model_checkpoint_paths: "/home/rer/Model/leftmodel/Matrixleftbi"\n')

    with open("./Model/rightmodel/checkpoint",'w') as f:
        f.write('model_checkpoint_path: "/home/rer/Model/rightmodel/Matrixrightbi"\n')
        f.write('all_model_checkpoint_paths: "/home/rer/Model/rightmodel/Matrixrightbi"\n')


    messagebox.showinfo("更换模型",
                        "更换模型用户的个人模型，请在一分钟后关闭程序")
    ssh = SSHClient()
    ssh.load_system_host_keys()
    # ssh.connect(hostname='192.168.43.23', username='rer',
    #             password='123')
    ssh.connect(hostname=ipaddress, username='rer',
                password='123')
    scp = SCPClient(ssh.get_transport())
    scp.put('./Model',recursive=True,
            remote_path='/home/rer/')
    # 添加一個i計時器然後直接結束
    time.sleep(10)
    messagebox.showinfo("完成模型替換",
                        "完成模型替換")
    scp.close()

    #   發送命令讓Matrix自動開始檢測
    # client.send('closeall'.encode())
    client.send('startpredict'.encode())
    showpredict = client.recv(1024).decode()
    if showpredict =='predictstarted':
        print("顯示預測結果")
        # sourceimg = cv2.imread("/home/gaofei/MAuto2-main-personal/MAuto2-main-master/Image/4lv9c8ee.png")
        head = cv2.imread("./Image/head.png")
        leftjaw = cv2.imread("./Image/leftjaw.png")
        rightjaw = cv2.imread("./Image/rightjaw.png")
        leftface = cv2.imread("./Image/leftface.png")
        rightface = cv2.imread("./Image/rightface.png")
        lefteye = cv2.imread("./Image/lefteye.png")
        righteye = cv2.imread("./Image/righteye.png")
        print("图片导入完成")

        # 可以添加一個案按鈕結束預測退出的過程
        while True:
            area = client.makefile().readline()
            if area == '0\n' or area == '4\n' or area == '8\n' or area == '12\n':
                print("當前在額頭區域")
                cv2.imshow('image', head)
                cv2.waitKey(10)
            elif area == '1\n' or area == '13\n':
                # print("current is" ,areaid)
                print("當前在左下頜線區域")
                cv2.imshow('image', leftjaw)
                cv2.waitKey(10)
            elif area == '5\n' or area == '9\n':
                print("當前在右下頜線區域")
                cv2.imshow('image', rightjaw)
                cv2.waitKey(10)
            elif area == '2\n' or area == '14\n':
                # print("current is" ,areaid)
                print("當前在左邊臉部")
                cv2.imshow('image', leftface)
                cv2.waitKey(10)
            elif area == '6\n' or area == '10\n':
                # print("current is" ,areaid)
                print("當前在右邊臉部")
                cv2.imshow('image', rightface)
                cv2.waitKey(10)
            elif area == '3\n' or area == '15\n':
                # print("current is" ,areaid)
                print("當前在左邊眼部")
                cv2.imshow('image', lefteye)
                cv2.waitKey(30)
            elif area == '7\n' or area == '11\n':
                # print("current is" ,areaid)
                print("當前在右邊眼部")
                cv2.imshow('image', righteye)
                cv2.waitKey(10)
            else:
                print("Matrix還沒開始運行預測")
            # print(area)






def SetCollectLeft():
    global getleftdatas
    if collectleft.get() == True:
        print("设置左边收集")
        getleftdatas = True
    else:
        print("设置左边不收集")
        getleftdatas = False

def SetCollectRight():
    global getrightdatas
    if collectright.get() == True:
        print("设置右边收集")
        getrightdatas = True
    else:
        print("设置右边不收集")
        getrightdatas = False

def TrainLeft():
    global trainLeft
    if trainleft.get() == True:
        print("设置左边训练")
        trainLeft = True
    else:
        print("设置左边不训练")
        trainLeft = False

def TrainRight():
    global trainRight
    if trainright.get() == True:
        print("设置右边训练")
        trainRight = True
    else:
        print("设置右边不收集")
        trainRight = False





if __name__=="__main__":
    # 主函数
    window = Tk()
    window.title("Matrix")
    window.geometry("400x200")
    collectleft = BooleanVar()
    collectright = BooleanVar()
    trainleft = BooleanVar()
    trainright = BooleanVar()

    connectMatrix = Button(window, text = "设置Matrix IP", command = connectMatrixCallBack)
    connectMatrix.place(x = 80,y = 20)

    connectMatrix = Button(window, text = "开始整体流程", command = StartFlow)
    connectMatrix.place(x = 230,y = 20)

    CollectLeftData = Checkbutton(window, text="收集左边脸部的数据", variable=collectleft, onvalue=True, offvalue=False,
                        command=SetCollectLeft)
    CollectLeftData.place(x = 50,y = 70)

    CollectRightData = Checkbutton(window, text="收集右边脸部的数据", variable=collectright, onvalue=True, offvalue=False,
                        command=SetCollectRight)
    CollectRightData.place(x = 200,y = 70)

    TrainLeft = Checkbutton(window, text="训练左边脸部模型", variable=trainleft, onvalue=True, offvalue=False,
                        command=TrainLeft)
    TrainLeft.place(x = 50,y = 120)

    TrainRight = Checkbutton(window, text="训练右边脸部模型", variable=trainright, onvalue=True, offvalue=False,
                        command=TrainRight)
    TrainRight.place(x = 200,y = 120)

    window.mainloop()




