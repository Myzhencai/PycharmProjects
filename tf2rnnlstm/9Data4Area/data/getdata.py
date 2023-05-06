import time
import serial
import threading
import numpy as np

# 用於保存數據的列表
datasaverleft = []
datasaverright = []

# 獲取左邊臉部數據
def serialleft(areaid,savetest,Savepath):
    serleft = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.01)
    serleft.flushInput()
    while True:
        currentdataleft = serleft.readline()
        if currentdataleft != b'':
            currentdataleft = str(currentdataleft, 'UTF-8')
            currentdatasaverleftleft = currentdataleft.split('\r\r\n')[0]
            currentdatasaverleftleft = currentdatasaverleftleft.split(",")
            dataarrayleft = np.array(currentdatasaverleftleft, dtype='float32').reshape((-1, 9))
            # 可能要枷鎖
            # datasaverleft.append(dataarrayleft[0][:9])
            # np.savetxt("./sensordataleft.txt", np.array(datasaverleft).reshape((-1, 9)))
            # print("dataarrayleft :", dataarrayleft)
            if areaid ==0:
                print("左邊額頭數據")
                areaarray = np.array([[1,0,0,0]])
                currentdataAndarea = np.c_[dataarrayleft,areaarray]
                datasaverleft.append(currentdataAndarea[0])
                dataarray = np.array(datasaverleft,dtype='float32').reshape((-1,13))
                # print(dataarray)
                if savetest:
                    np.savetxt(Savepath+"area{0}test.txt".format(areaid),dataarray)
                else:
                    np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)
            elif areaid ==1:
                print("對應左下頜")
                areaarray = np.array([[0, 1, 0, 0]])
                currentdataAndarea = np.c_[dataarrayleft, areaarray]
                datasaverleft.append(currentdataAndarea[0])
                dataarray = np.array(datasaverleft, dtype='float32').reshape((-1, 13))
                if savetest:
                    np.savetxt(Savepath + "area{0}test.txt".format(areaid), dataarray)
                else:
                    np.savetxt(Savepath + "area{0}.txt".format(areaid), dataarray)
            elif areaid ==2:
                print("對應左邊面部")
                areaarray = np.array([[0, 0, 1, 0]])
                currentdataAndarea = np.c_[dataarrayleft, areaarray]
                datasaverleft.append(currentdataAndarea[0])
                dataarray = np.array(datasaverleft, dtype='float32').reshape((-1, 13))
                if savetest:
                    np.savetxt(Savepath + "area{0}test.txt".format(areaid), dataarray)
                else:
                    np.savetxt(Savepath + "area{0}.txt".format(areaid), dataarray)
            elif areaid ==3:
                print("左邊眼周")
                areaarray = np.array([[0, 0, 0, 1]])
                currentdataAndarea = np.c_[dataarrayleft, areaarray]
                datasaverleft.append(currentdataAndarea[0])
                dataarray = np.array(datasaverleft, dtype='float32').reshape((-1, 13))
                if savetest:
                    np.savetxt(Savepath + "area{0}test.txt".format(areaid), dataarray)
                else:
                    np.savetxt(Savepath + "area{0}.txt".format(areaid), dataarray)
        # else:
        #     print("no data")



# 獲取右邊臉部數據
def serialrifht(areaid,savetest,Savepath):
    serringht = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.01)
    serringht.flushInput()
    while True:
        currentdataright = serringht.readline()
        if currentdataright != b'':
            currentdataright = str(currentdataright, 'UTF-8')
            currentdatasaverleftright = currentdataright.split('\r\n')[0]
            currentdatasaverleftright = currentdatasaverleftright.split(",")
            dataarrayright = np.array(currentdatasaverleftright, dtype='float32').reshape((-1, 9))
            # datasaverright.append(dataarrayright[0][:9])
            # np.savetxt("./sensordataright.txt", np.array(datasaverright).reshape((-1, 9)))
            # print("dataarrayright :", dataarrayright)
            if areaid ==0:
                print("右邊額頭數據")
                areaarray = np.array([[1,0,0,0]])
                currentdataAndarea = np.c_[dataarrayright,areaarray]
                datasaverright.append(currentdataAndarea[0])
                dataarray = np.array(datasaverright,dtype='float32').reshape((-1,13))
                if savetest:
                    np.savetxt(Savepath+"area{0}test.txt".format(areaid),dataarray)
                else:
                    np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)

            if areaid ==1:
                print("對應右邊下頜")
                areaarray = np.array([[0,1,0,0]])
                currentdataAndarea = np.c_[dataarrayright,areaarray]
                datasaverright.append(currentdataAndarea[0])
                dataarray = np.array(datasaverright,dtype='float32').reshape((-1,13))
                if savetest:
                    np.savetxt(Savepath+"area{0}test.txt".format(areaid),dataarray)
                else:
                    np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)

            if areaid ==2:
                print("右邊面部數據")
                areaarray = np.array([[0,0,1,0]])
                currentdataAndarea = np.c_[dataarrayright,areaarray]
                datasaverright.append(currentdataAndarea[0])
                dataarray = np.array(datasaverright,dtype='float32').reshape((-1,13))
                if savetest:
                    np.savetxt(Savepath+"area{0}test.txt".format(areaid),dataarray)
                else:
                    np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)

            if areaid ==3:
                print("右邊眼周")
                areaarray = np.array([[0,0,0,1]])
                currentdataAndarea = np.c_[dataarrayright,areaarray]
                datasaverright.append(currentdataAndarea[0])
                dataarray = np.array(datasaverright,dtype='float32').reshape((-1,13))
                if savetest:
                    np.savetxt(Savepath+"area{0}test.txt".format(areaid),dataarray)
                else:
                    np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)


if __name__ == '__main__':
    # 0 對應額頭 ,1 對應左下頜,2對應右下頜,3左邊臉部,4對應右邊臉部,5對應左邊眼周,6對應右邊臉周
    area = 3
    test = False
    leftsavepath = "./left/"
    rithtsavepath = "./right/"

    serialleft(area, test, leftsavepath)
    # serialrifht(area, test, rithtsavepath)


    # 开启线程
    # all_thread.start()
    # right_thread.start()