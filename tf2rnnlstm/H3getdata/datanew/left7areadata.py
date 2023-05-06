import time
import serial
import threading
import numpy as np

# 用於保存數據的列表
datasaverleft = []
datasaverright = []

# 獲取左邊臉部數據
def serialsave(areaid,savetest,Savepathleft,Savepathright):
    serleft = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.01)
    serleft.flushInput()
    while True:
        currentdataleft = serleft.readline()
        if currentdataleft != b'':
            currentdataleft = str(currentdataleft, 'UTF-8')
            currentdatasaverleftleft = currentdataleft.split('\r\n')[0]
            # print(currentdatasaverleftleft.split(","))
            templist = currentdatasaverleftleft.split(",")
            sideparm = templist.pop(0)
            currentdatasaverleftleft = templist

            dataarrayleft = np.array(currentdatasaverleftleft, dtype='float32').reshape((-1, 9))
            # 可能要枷鎖
            # datasaverleft.append(dataarrayleft[0][:9])
            # np.savetxt("./sensordataleft.txt", np.array(datasaverleft).reshape((-1, 9)))
            # print("dataarrayleft :", dataarrayleft)
            if areaid ==0:
                print("左邊額頭數據")
                areaarray = np.array([[1,0,0,0,0,0,0]])
                currentdataAndarea = np.c_[dataarrayleft,areaarray]
                datasaverleft.append(currentdataAndarea[0])
                dataarray = np.array(datasaverleft,dtype='float32').reshape((-1,16))
                if savetest:
                    if sideparm=='0':
                        np.savetxt(Savepathleft+"area{0}test.txt".format(areaid),dataarray)
                    else:
                        np.savetxt(Savepathright + "area{0}test.txt".format(areaid), dataarray)
                else:
                    if sideparm=='0':
                        np.savetxt(Savepathleft+"area{0}.txt".format(areaid),dataarray)
                    else:
                        np.savetxt(Savepathright + "area{0}.txt".format(areaid), dataarray)
            elif areaid ==1:
                print("對應左下頜")
                areaarray = np.array([[0, 1, 0, 0, 0, 0, 0]])
                currentdataAndarea = np.c_[dataarrayleft, areaarray]
                datasaverleft.append(currentdataAndarea[0])
                dataarray = np.array(datasaverleft, dtype='float32').reshape((-1, 16))
                if savetest:
                    if sideparm == '0':
                        np.savetxt(Savepathleft + "area{0}test.txt".format(areaid), dataarray)
                    else:
                        np.savetxt(Savepathright + "area{0}test.txt".format(areaid), dataarray)
                else:
                    if sideparm == '0':
                        np.savetxt(Savepathleft + "area{0}.txt".format(areaid), dataarray)
                    else:
                        np.savetxt(Savepathright + "area{0}.txt".format(areaid), dataarray)
            elif areaid ==2:
                print("2對應右下頜")
                areaarray = np.array([[0, 0, 1, 0,0,0,0]])
                currentdataAndarea = np.c_[dataarrayleft, areaarray]
                datasaverleft.append(currentdataAndarea[0])
                dataarray = np.array(datasaverleft, dtype='float32').reshape((-1, 16))
                if savetest:
                    if sideparm == '0':
                        np.savetxt(Savepathleft + "area{0}test.txt".format(areaid), dataarray)
                    else:
                        np.savetxt(Savepathright + "area{0}test.txt".format(areaid), dataarray)
                else:
                    if sideparm == '0':
                        # print("left", currentdataAndarea)
                        np.savetxt(Savepathleft + "area{0}.txt".format(areaid), dataarray)
                    else:
                        # print("right", currentdataAndarea)
                        np.savetxt(Savepathright + "area{0}.txt".format(areaid), dataarray)

            elif areaid ==3:
                print("左邊臉部")
                areaarray = np.array([[0, 0, 0, 1,0,0,0]])
                currentdataAndarea = np.c_[dataarrayleft, areaarray]
                datasaverleft.append(currentdataAndarea[0])
                dataarray = np.array(datasaverleft, dtype='float32').reshape((-1, 16))

                if savetest:
                    if sideparm == '0':
                        np.savetxt(Savepathleft + "area{0}test.txt".format(areaid), dataarray)
                    else:
                        np.savetxt(Savepathright + "area{0}test.txt".format(areaid), dataarray)
                else:
                    if sideparm == '0':
                        # print("left",currentdataAndarea)
                        np.savetxt(Savepathleft + "area{0}.txt".format(areaid), dataarray)
                    else:
                        # print("right", currentdataAndarea)
                        np.savetxt(Savepathright + "area{0}.txt".format(areaid), dataarray)

            elif areaid ==4:
                print("對應右邊臉部")
                areaarray = np.array([[0, 0, 0, 0,1,0,0]])
                currentdataAndarea = np.c_[dataarrayleft, areaarray]
                datasaverleft.append(currentdataAndarea[0])
                dataarray = np.array(datasaverleft, dtype='float32').reshape((-1, 16))
                if savetest:
                    if sideparm == '0':
                        np.savetxt(Savepathleft + "area{0}test.txt".format(areaid), dataarray)
                    else:
                        np.savetxt(Savepathright + "area{0}test.txt".format(areaid), dataarray)
                else:
                    if sideparm == '0':
                        np.savetxt(Savepathleft + "area{0}.txt".format(areaid), dataarray)
                    else:
                        np.savetxt(Savepathright + "area{0}.txt".format(areaid), dataarray)

            elif areaid ==5:
                print("對應左邊眼周")
                areaarray = np.array([[0, 0, 0, 0,0,1,0]])
                currentdataAndarea = np.c_[dataarrayleft, areaarray]
                datasaverleft.append(currentdataAndarea[0])
                dataarray = np.array(datasaverleft, dtype='float32').reshape((-1, 16))
                if savetest:
                    if sideparm == '0':
                        np.savetxt(Savepathleft + "area{0}test.txt".format(areaid), dataarray)
                    else:
                        np.savetxt(Savepathright + "area{0}test.txt".format(areaid), dataarray)
                else:
                    if sideparm == '0':
                        np.savetxt(Savepathleft + "area{0}.txt".format(areaid), dataarray)
                    else:
                        np.savetxt(Savepathright + "area{0}.txt".format(areaid), dataarray)
            elif areaid ==6:
                print("對應右邊臉周")
                areaarray = np.array([[0, 0, 0, 0,0,0,1]])
                currentdataAndarea = np.c_[dataarrayleft, areaarray]
                datasaverleft.append(currentdataAndarea[0])
                dataarray = np.array(datasaverleft, dtype='float32').reshape((-1, 16))
                if savetest:
                    if sideparm == '0':
                        np.savetxt(Savepathleft + "area{0}test.txt".format(areaid), dataarray)
                    else:
                        np.savetxt(Savepathright + "area{0}test.txt".format(areaid), dataarray)
                else:
                    if sideparm == '0':
                        np.savetxt(Savepathleft + "area{0}.txt".format(areaid), dataarray)
                    else:
                        np.savetxt(Savepathright + "area{0}.txt".format(areaid), dataarray)
        # else:
        #     print("no data")


if __name__ == '__main__':
    # 0 對應額頭 ,1 對應左下頜,2對應右下頜,3左邊臉部,4對應右邊臉部,5對應左邊眼周,6對應右邊臉周
    area = 6
    # test = True
    test = False
    leftsavepath = "./left/"
    rithtsavepath = "./ritht/"

    serialsave(area, test, leftsavepath,rithtsavepath)


    # 开启线程
    # all_thread.start()
    # right_thread.start()


