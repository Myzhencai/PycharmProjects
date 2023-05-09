import time
import serial
import threading
import numpy as np
import os
import cv2
from tqdm import tqdm
# 用於保存數據的列表
datasaverleft = []
datasaverright = []

# 獲取左邊臉部數據
def serialleft(areaid,savetest,Savepath,dataenoughnum,serialport):
    datasaverleft.clear()
    serleft = serial.Serial(serialport, 115200, timeout=0.01)
    # serleft.open()
    serleft.flushInput()
    currentnum = 0
    while currentnum < dataenoughnum:
        currentdataleft = serleft.readline()
        if currentdataleft != b'':
            currentdataleft = str(currentdataleft, 'UTF-8')
            currentdatasaverleftleftold = currentdataleft.split('\r\n')[0]
            currentdatasaverleftleft = currentdatasaverleftleftold.split(",")
            inversecurrentdatasaverleftleft =currentdatasaverleftleftold.split(",")
            inversecurrentdatasaverleftleft.reverse()
            # print("currentdatasaverleftleft",currentdatasaverleftleft)
            # print("inverse", inversecurrentdatasaverleftleft)
            currentdatasaverleftleft = currentdatasaverleftleft + inversecurrentdatasaverleftleft
            # print(currentdatasaverleftleft)
            # currentdatasaverleftleft = currentdatasaverleftleft+inversecurrent
            dataarrayleft = np.array(currentdatasaverleftleft, dtype='float32').reshape((-1, 18))
            # 可能要枷鎖
            # datasaverleft.append(dataarrayleft[0][:9])
            # np.savetxt("./sensordataleft.txt", np.array(datasaverleft).reshape((-1, 9)))
            # print("dataarrayleft :", dataarrayleft)
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
        #     print("no data")
    # serleft.close()
    return areaid



# 獲取右邊臉部數據
def serialrifht(areaid,savetest,Savepath,dataenoughnum,serialport):
    datasaverright.clear()
    serringht = serial.Serial(serialport, 115200, timeout=0.01)
    # serringht.open()
    serringht.flushInput()
    currentnum = 0
    while currentnum < dataenoughnum:
        # print("current num ",currentnum)
        currentdataright = serringht.readline()
        if currentdataright != b'':
            currentdataright = str(currentdataright, 'UTF-8')
            currentdatasaverleftrightold = currentdataright.split('\r\n')[0]
            currentdatasaverleftright = currentdatasaverleftrightold.split(",")
            inversecurrentdatasaverright = currentdatasaverleftrightold.split(",")
            inversecurrentdatasaverright.reverse()
            # print("currentdatasaverleftleft",currentdatasaverleftleft)
            # print("inverse", inversecurrentdatasaverleftleft)
            currentdatasaverleftleft = currentdatasaverleftright + inversecurrentdatasaverright
            # print(currentdatasaverleftleft)
            # currentdatasaverleftleft = currentdatasaverleftleft+inversecurrent
            dataarrayright = np.array(currentdatasaverleftleft, dtype='float32').reshape((-1, 18))
            # dataarrayright = np.array(currentdatasaverleftright, dtype='float32').reshape((-1, 9))
            # datasaverright.append(dataarrayright[0][:9])
            # np.savetxt("./sensordataright.txt", np.array(datasaverright).reshape((-1, 9)))
            # print("dataarrayright :", dataarrayright)
            if areaid ==0:
                print("右邊額頭數據", currentnum)
                areaarray = np.array([[1,0,0,0]])
                currentdataAndarea = np.c_[dataarrayright,areaarray]
                datasaverright.append(currentdataAndarea[0])
                dataarray = np.array(datasaverright,dtype='float32').reshape((-1,22))
                if savetest:
                    np.savetxt(Savepath+"area{0}test.txt".format(areaid),dataarray)
                else:
                    np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)
                currentnum += 1
            elif areaid ==1:
                print("對應右邊下頜", currentnum)
                areaarray = np.array([[0,1,0,0]])
                currentdataAndarea = np.c_[dataarrayright,areaarray]
                datasaverright.append(currentdataAndarea[0])
                dataarray = np.array(datasaverright,dtype='float32').reshape((-1,22))
                if savetest:
                    np.savetxt(Savepath+"area{0}test.txt".format(areaid),dataarray)
                else:
                    np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)
                currentnum += 1
            elif areaid ==2:
                print("右邊面部數據", currentnum)
                areaarray = np.array([[0,0,1,0]])
                currentdataAndarea = np.c_[dataarrayright,areaarray]
                datasaverright.append(currentdataAndarea[0])
                dataarray = np.array(datasaverright,dtype='float32').reshape((-1,22))
                if savetest:
                    np.savetxt(Savepath+"area{0}test.txt".format(areaid),dataarray)
                else:
                    np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)
                currentnum += 1
            elif areaid ==3:
                print("右邊眼周", currentnum)
                areaarray = np.array([[0,0,0,1]])
                currentdataAndarea = np.c_[dataarrayright,areaarray]
                datasaverright.append(currentdataAndarea[0])
                dataarray = np.array(datasaverright,dtype='float32').reshape((-1,22))
                if savetest:
                    np.savetxt(Savepath+"area{0}test.txt".format(areaid),dataarray)
                else:
                    np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)
                currentnum += 1
    # serringht.close()
    return areaid


if __name__ == '__main__':
    # 0 對應額頭 ,1 對應左下頜,2對應右下頜,3左邊臉部,4對應右邊臉部,5對應左邊眼周,6對應右邊臉周
    serialport = "/dev/ttyUSB0"
    arealist = [0,1,2,3]
    savetest = False
    dataenoughnum = 2000
    savePathlist = ["./personaldata/Soarleftbi/","./personaldata/Soarrightbi/"]

    for savepath in savePathlist:
        dir = savepath
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))

    for i in range(len(savePathlist)):
        if i == 0 :
            for areaid in arealist:
                if areaid == 0:
                    while input("請將美容儀器開機放置在額頭區域（左），完成後請按Enter鍵 :") !='':
                          print("重新輸入")
                    print("左邊額頭數據")
                    finishedid  = serialleft(areaid, savetest, savePathlist[0], dataenoughnum,serialport)
                    # print(finishedid)
                if areaid == 1 and finishedid==0:
                    while input("請將美容儀器開機放置在下頜線區域（左），完成後請按Enter鍵 :") !='':
                          print("重新輸入")
                    print("對應左下頜")
                    finishedid  =serialleft(areaid, savetest, savePathlist[0], dataenoughnum,serialport)
                    # print(finishedid)
                if areaid == 2 and finishedid==1:
                    while input("請將美容儀器開機放置在臉部區域（左），完成後請按Enter鍵 :") !='':
                          print("重新輸入")
                    print("對應左邊面部")
                    finishedid  =serialleft(areaid, savetest, savePathlist[0], dataenoughnum,serialport)
                if areaid == 3 and finishedid==2:
                    while input("請將美容儀器開機放置在眼周區域（左），完成後請按Enter鍵 :") !='':
                          print("重新輸入")
                    print("左邊眼周")
                    finishedid  =serialleft(areaid, savetest, savePathlist[0], dataenoughnum,serialport)
                if finishedid ==3:
                    print("完成左邊臉部的數據採集")
        # if i == 1:
        #     for areaid in arealist:
        #         if areaid == 0:
        #             while input("請將美容儀器開機放置在額頭區域（右），完成後請按Enter鍵 :") != '':
        #                 print("重新輸入")
        #             print("右邊額頭數據")
        #             finishedid = serialrifht(areaid, savetest, savePathlist[1], dataenoughnum, serialport)
        #             print(finishedid)
        #         if areaid == 1 and finishedid == 0:
        #             while input("請將美容儀器開機放置在下頜線區域（右），完成後請按Enter鍵 :") != '':
        #                 print("重新輸入")
        #             print("對應右邊下頜")
        #             finishedid = serialrifht(areaid, savetest, savePathlist[1], dataenoughnum, serialport)
        #         if areaid == 2 and finishedid == 1:
        #             while input("請將美容儀器開機放置在臉部區域（右），完成後請按Enter鍵 :") != '':
        #                 print("重新輸入")
        #             print("右邊面部數據")
        #             finishedid = serialrifht(areaid, savetest, savePathlist[1], dataenoughnum, serialport)
        #         if areaid == 3 and finishedid == 2:
        #             while input("請將美容儀器開機放置在眼周區域（右），完成後請按Enter鍵 :") != '':
        #                 print("重新輸入")
        #             print("右邊眼周")
        #             finishedid = serialrifht(areaid, savetest, savePathlist[1], dataenoughnum, serialport)
        #         if finishedid == 3:
        #             print("完成右邊臉部的數據採集")


    # area = 3
    # test = False
    # leftsavepath = "./Soarleftbi/"
    # rithtsavepath = "./Soarrightbi/"

    # serialleft(area, test, leftsavepath)
    # serialrifht(area, test, rithtsavepath)

    # 开启线程
    # all_thread.start()
    # right_thread.start()