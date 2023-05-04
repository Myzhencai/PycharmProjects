import numpy as np
import serial
import time

# 左邊
serleft = serial.Serial("/dev/ttyS2",115200,timeout = 0.01)
serleft.flushInput()
# 右邊
serringht = serial.Serial("/dev/ttyS1",115200,timeout = 0.01)
serringht.flushInput()

datasaverleft = []
datasaverright = []
datalist = []
def main(areaid,savetest,Savepath):
    while True:
        # 获取左右串口缓冲区数据
        currentdataleft = serleft.readline()
        currentdataright = serringht.readline()
        # 先獲得左右兩邊的數據並確定需要保存哪一個
        if currentdataleft != b'' and currentdataright != b'':
            # starttime= time.time()
            # 左邊數據
            currentdataleft = str(currentdataleft, 'UTF-8')
            currentdatalistleft = currentdataleft.split('\n')[0]
            currentdatalistleft = currentdatalistleft.split(",")
            dataarrayleft = np.array(currentdatalistleft, dtype='float16').reshape((-1, 9))
            datasumleft = np.sum(dataarrayleft)
            # 右邊數據
            currentdataright = str(currentdataright, 'UTF-8')
            currentdatalistright = currentdataright.split('\n')[0]
            currentdatalistright = currentdatalistright.split(",")
            dataarrayright = np.array(currentdatalistright, dtype='float16').reshape((-1, 9))
            datasumright = np.sum(dataarrayright)
            # 判定選擇那一個
            if datasumleft > datasumright:
                currentdata = dataarrayleft
                print("dataarrayleft :", dataarrayleft)
                print("dataarrayright :", dataarrayright)
                print("currentdata chosen : left", )
            else:
                currentdata = dataarrayright
                print("dataarrayleft :", dataarrayleft)
                print("dataarrayright :", dataarrayright)
                print("currentdata chosen : right", )

        elif currentdataleft != b'' and currentdataright == b'':
            currentdataleft = str(currentdataleft, 'UTF-8')
            currentdatalistleft = currentdataleft.split('\n')[0]
            currentdatalistleft = currentdatalistleft.split(",")
            dataarrayleft = np.array(currentdatalistleft, dtype='float16').reshape((-1, 9))
            currentdata = dataarrayleft
            print("dataarrayleft :", dataarrayleft)
            print("dataarrayright : no data")
            print("currentdata chosen : left")
        elif currentdataleft == b'' and currentdataright != b'':
            currentdataright = str(currentdataright, 'UTF-8')
            currentdatalistright = currentdataright.split('\n')[0]
            currentdatalistright = currentdatalistright.split(",")
            dataarrayright = np.array(currentdatalistright, dtype='float16').reshape((-1, 9))
            currentdata = dataarrayright
            print("dataarrayleft : no data")
            print("dataarrayright :",dataarrayright)
            print("currentdata chosen : right" )


        # if currentdataright != b'':
        #     currentdataright = str(currentdataright, 'UTF-8')
        #     currentdatalistright = currentdataright.split('\n')[0]
        #     currentdatalistright = currentdatalistright.split(",")
        #     dataarrayright = np.array(currentdatalistright, dtype='float16').reshape((-1, 9))
        #     datasaverright.append(dataarrayright[0][:9])
        #     np.savetxt("./sensordataright.txt", np.array(datasaverright).reshape((-1, 9)))
        #     print("dataarrayright :", dataarrayright)
        #
        #
        # if currentdata !=b'' and currentdata !=b'\n':
        #     if areaid ==0:
        #         print("額頭數據")
        #         currentdata = str(currentdata, 'UTF-8')
        #         currentdatalist = currentdata.split('\r\n')[0]
        #         currentdatalist = currentdatalist.split(",")+[1,0,0,0,0,0,0]
        #         datalist.append(currentdatalist)
        #         print(currentdata)
        #         dataarray = np.array(datalist,dtype='float32').reshape((-1,25))
        #         if savetest:
        #             np.savetxt(Savepath+"area{0}test.txt".format(areaid),dataarray)
        #         else:
        #             np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)
        #     elif areaid ==1:
        #         print("對應左下頜")
        #         currentdata = str(currentdata, 'UTF-8')
        #         currentdatalist = currentdata.split('\r\n')[0]
        #         currentdatalist = currentdatalist.split(",")+[0,1,0,0,0,0,0]
        #         datalist.append(currentdatalist)
        #         print(currentdata)
        #         dataarray = np.array(datalist,dtype='float32').reshape((-1,25))
        #         if savetest:
        #             np.savetxt(Savepath+"area{0}test.txt".format(areaid), dataarray)
        #         else:
        #             np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)
        #     elif areaid ==2:
        #         print("對應右下頜")
        #         currentdata = str(currentdata, 'UTF-8')
        #         currentdatalist = currentdata.split('\r\n')[0]
        #         currentdatalist = currentdatalist.split(",")+[0,0,1,0,0,0,0]
        #         datalist.append(currentdatalist)
        #         print(currentdata)
        #         dataarray = np.array(datalist,dtype='float32').reshape((-1,25))
        #         if savetest:
        #             np.savetxt(Savepath+"area{0}test.txt".format(areaid), dataarray)
        #         else:
        #             np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)
        #     elif areaid ==3:
        #         print("左邊臉部")
        #         currentdata = str(currentdata, 'UTF-8')
        #         currentdatalist = currentdata.split('\r\n')[0]
        #         currentdatalist = currentdatalist.split(",")+[0,0,0,1,0,0,0]
        #         datalist.append(currentdatalist)
        #         print(currentdata)
        #         dataarray = np.array(datalist,dtype='float32').reshape((-1,25))
        #         if savetest:
        #             np.savetxt(Savepath+"area{0}test.txt".format(areaid), dataarray)
        #         else:
        #             np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)
        #     elif areaid ==4:
        #         print("對應右邊臉部")
        #         currentdata = str(currentdata, 'UTF-8')
        #         currentdatalist = currentdata.split('\r\n')[0]
        #         currentdatalist = currentdatalist.split(",")+[0,0,0,0,1,0,0]
        #         datalist.append(currentdatalist)
        #         print(currentdata)
        #         dataarray = np.array(datalist,dtype='float32').reshape((-1,25))
        #         if savetest:
        #             np.savetxt(Savepath+"area{0}test.txt".format(areaid), dataarray)
        #         else:
        #             np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)
        #     elif areaid ==5:
        #         print("對應左邊眼周")
        #         currentdata = str(currentdata, 'UTF-8')
        #         currentdatalist = currentdata.split('\r\n')[0]
        #         currentdatalist = currentdatalist.split(",")+[0,0,0,0,0,1,0]
        #         datalist.append(currentdatalist)
        #         print(currentdata)
        #         dataarray = np.array(datalist,dtype='float32').reshape((-1,25))
        #         if savetest:
        #             np.savetxt(Savepath+"area{0}test.txt".format(areaid), dataarray)
        #         else:
        #             np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)
        #     elif areaid ==6:
        #         print("對應右邊臉周")
        #         currentdata = str(currentdata, 'UTF-8')
        #         currentdatalist = currentdata.split('\r\n')[0]
        #         currentdatalist = currentdatalist.split(",")+[0,0,0,0,0,0,1]
        #         datalist.append(currentdatalist)
        #         print(currentdata)
        #         dataarray = np.array(datalist,dtype='float32').reshape((-1,25))
        #         if savetest:
        #             np.savetxt(Savepath+"area{0}test.txt".format(areaid), dataarray)
        #         else:
        #             np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)
        time.sleep(0.001) # 延时0.1秒，免得CPU出问题

if __name__ == '__main__':
    # 0 對應額頭 ,1 對應左下頜,2對應右下頜,3左邊臉部,4對應右邊臉部,5對應左邊眼周,6對應右邊臉周
    area = 6
    savepath = "/home/gaofei/PycharmProjects/ElectroMagnetArea/soarrealtimedatafortrain/"
    # test = True
    test = False
    main(area,test,savepath)
