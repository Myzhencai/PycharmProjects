import time
import serial
import threading
import numpy as np

#共享的buffer內存數據
realtimebuffer = [None,None]
# 用於保存數據的列表
datalist =[]

# 獲取左邊臉部數據
def serialleft():
    serleft = serial.Serial("/dev/ttyS2", 115200, timeout=0.01)
    serleft.flushInput()
    while True:
        currentdataleft = serleft.readline()
        if currentdataleft != b'':
            currentdataleft = str(currentdataleft, 'UTF-8')
            currentdatalistleft = currentdataleft.split('\r\n')[0]
            currentdatalistleft = currentdatalistleft.split(",")
            dataarrayleft = np.array(currentdatalistleft, dtype='float32').reshape((-1, 9))
            # 可能要枷鎖
            realtimebuffer[0] = dataarrayleft
            # print(dataarrayleft)
        else:
            realtimebuffer[0] = None

# 獲取右邊臉部數據
def serialrifht():
    serringht = serial.Serial("/dev/ttyS1", 115200, timeout=0.01)
    serringht.flushInput()
    while True:
        currentdataright = serringht.readline()
        if currentdataright != b'':
            currentdataright = str(currentdataright, 'UTF-8')
            currentdatalistright = currentdataright.split('\r\n')[0]
            currentdatalistright = currentdatalistright.split(",")
            dataarrayright = np.array(currentdatalistright, dtype='float32').reshape((-1, 9))
            realtimebuffer[1] = dataarrayright
        else:
            realtimebuffer[1] = None

# 從共享內存裏提取數據
def chosendata(areaid,savetest,Savepath):
    chosendataresult = np.zeros((1, 9))
    while True:
        if (realtimebuffer[0] is not None) and (realtimebuffer[1] is not None):
            # leftsum = np.sum(realtimebuffer[0]*realtimebuffer[0])
            # rightsum = np.sum(realtimebuffer[1]*realtimebuffer[1])
            leftsum =abs(realtimebuffer[0][0][0])
            rightsum = abs(realtimebuffer[1][0][0])
            if leftsum > rightsum:
                chosendataresult = realtimebuffer[0]
                # datalist.append(chosendataresult[0][:9])
                # print("chose one left")
        #         print(chosendataresult)
            else:
                chosendataresult = realtimebuffer[1]
                # datalist.append(chosendataresult[0][:9])
                # print("chose one right")
        #         print(chosendataresult)
        elif (realtimebuffer[0] is None) and (realtimebuffer[1] is not None):
            chosendataresult = realtimebuffer[1]
            # datalist.append(chosendataresult[0][:9])
            # print("chose 2 right")
            # print(chosendataresult)
        elif (realtimebuffer[0] is not None) and (realtimebuffer[1] is None):
            chosendataresult = realtimebuffer[0]
            # datalist.append(chosendataresult[0][:9])
            # print("chose 2 left")
            # print(chosendataresult)
        # 只有有數據才保存
        else:
            continue
        # 根據實際的分區和保存地址保存選擇出來的數據
        if chosendataresult is not None:
            if areaid ==0:
                print("額頭數據")
                areaarray = np.array([[1,0,0,0,0,0,0]])
                currentdataAndarea = np.c_[chosendataresult,areaarray]
                datalist.append(currentdataAndarea[0])
                dataarray = np.array(datalist,dtype='float32').reshape((-1,16))
                if savetest:
                    np.savetxt(Savepath+"area{0}test.txt".format(areaid),dataarray)
                else:
                    np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)
            elif areaid ==1:
                print("對應左下頜")
                areaarray = np.array([[0, 1, 0, 0, 0, 0, 0]])
                currentdataAndarea = np.c_[chosendataresult, areaarray]
                datalist.append(currentdataAndarea[0])
                dataarray = np.array(datalist, dtype='float32').reshape((-1, 16))
                if savetest:
                    np.savetxt(Savepath + "area{0}test.txt".format(areaid), dataarray)
                else:
                    np.savetxt(Savepath + "area{0}.txt".format(areaid), dataarray)
            elif areaid ==2:
                print("對應右下頜")
                areaarray = np.array([[0, 0, 1, 0, 0, 0, 0]])
                currentdataAndarea = np.c_[chosendataresult, areaarray]
                datalist.append(currentdataAndarea[0])
                dataarray = np.array(datalist, dtype='float32').reshape((-1, 16))
                if savetest:
                    np.savetxt(Savepath + "area{0}test.txt".format(areaid), dataarray)
                else:
                    np.savetxt(Savepath + "area{0}.txt".format(areaid), dataarray)
            elif areaid ==3:
                print("左邊臉部")
                areaarray = np.array([[0, 0, 0, 1, 0, 0, 0]])
                currentdataAndarea = np.c_[chosendataresult, areaarray]
                datalist.append(currentdataAndarea[0])
                dataarray = np.array(datalist, dtype='float32').reshape((-1, 16))
                if savetest:
                    np.savetxt(Savepath + "area{0}test.txt".format(areaid), dataarray)
                else:
                    np.savetxt(Savepath + "area{0}.txt".format(areaid), dataarray)
            elif areaid ==4:
                print("對應右邊臉部")
                areaarray = np.array([[0, 0, 0, 0, 1, 0, 0]])
                currentdataAndarea = np.c_[chosendataresult, areaarray]
                datalist.append(currentdataAndarea[0])
                dataarray = np.array(datalist, dtype='float32').reshape((-1, 16))
                if savetest:
                    np.savetxt(Savepath + "area{0}test.txt".format(areaid), dataarray)
                else:
                    np.savetxt(Savepath + "area{0}.txt".format(areaid), dataarray)
            elif areaid ==5:
                print("對應左邊眼周")
                areaarray = np.array([[0, 0, 0, 0, 0, 1, 0]])
                currentdataAndarea = np.c_[chosendataresult, areaarray]
                datalist.append(currentdataAndarea[0])
                dataarray = np.array(datalist, dtype='float32').reshape((-1, 16))
                if savetest:
                    np.savetxt(Savepath + "area{0}test.txt".format(areaid), dataarray)
                else:
                    np.savetxt(Savepath + "area{0}.txt".format(areaid), dataarray)
            elif areaid ==6:
                print("對應右邊臉周")
                areaarray = np.array([[0, 0, 0, 0, 0, 0, 1]])
                currentdataAndarea = np.c_[chosendataresult, areaarray]
                datalist.append(currentdataAndarea[0])
                dataarray = np.array(datalist, dtype='float32').reshape((-1, 16))
                if savetest:
                    np.savetxt(Savepath + "area{0}test.txt".format(areaid), dataarray)
                else:
                    np.savetxt(Savepath + "area{0}.txt".format(areaid), dataarray)
        # time.sleep(0.001)




if __name__ == '__main__':
    # 0 對應額頭 ,1 對應左下頜,2對應右下頜,3左邊臉部,4對應右邊臉部,5對應左邊眼周,6對應右邊臉周
    area = 0
    # test = True
    test = False
    savepath = "./7Areadata/"

    left_thread = threading.Thread(target=serialleft)
    right_thread = threading.Thread(target=serialrifht)
    chosen_thread = threading.Thread(target=chosendata,args=(area,test,savepath,))

    # 开启线程
    left_thread.start()
    right_thread.start()
    chosen_thread.start()
