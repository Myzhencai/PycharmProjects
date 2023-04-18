import numpy as np
import serial # 导入串口包
import time # 导入时间包

ser = serial.Serial("/dev/ttyUSB0",256000,timeout = 0.01) # 开启com3口，波特率115200，超时5

ser.flushInput() # 清空缓冲区

def main(areaid,savetest,Savepath):
    datalist = []
    while True:
        currentdata = ser.readline() # 获取串口缓冲区数据
        if currentdata !=b'' :
            if areaid ==0:
                print("額頭數據")
                currentdata = str(currentdata, 'UTF-8')
                currentdatalist = currentdata.split('\r\n')[0]
                currentdatalist = currentdatalist.split(",")+[1,0,0,0]
                datalist.append(currentdatalist)
                print(currentdata)
                dataarray = np.array(datalist,dtype='float32').reshape((-1,13))
                if savetest:
                    np.savetxt(Savepath+"area{0}test.txt".format(areaid),dataarray)
                else:
                    np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)
            elif areaid ==1:
                print("下頜線數據")
                currentdata = str(currentdata, 'UTF-8')
                currentdatalist = currentdata.split('\r\n')[0]
                currentdatalist = currentdatalist.split(",")+[0,1,0,0]
                datalist.append(currentdatalist)
                print(currentdata)
                dataarray = np.array(datalist,dtype='float32').reshape((-1,13))
                if savetest:
                    np.savetxt(Savepath+"area{0}test.txt".format(areaid), dataarray)
                else:
                    np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)
            elif areaid ==2:
                print("面部數據")
                currentdata = str(currentdata, 'UTF-8')
                currentdatalist = currentdata.split('\r\n')[0]
                currentdatalist = currentdatalist.split(",")+[0,0,1,0]
                datalist.append(currentdatalist)
                print(currentdata)
                dataarray = np.array(datalist,dtype='float32').reshape((-1,13))
                if savetest:
                    np.savetxt(Savepath+"area{0}test.txt".format(areaid), dataarray)
                else:
                    np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)
            elif areaid ==3:
                print("眼周數據")
                currentdata = str(currentdata, 'UTF-8')
                currentdatalist = currentdata.split('\r\n')[0]
                currentdatalist = currentdatalist.split(",")+[0,0,0,1]
                datalist.append(currentdatalist)
                print(currentdata)
                dataarray = np.array(datalist,dtype='float32').reshape((-1,13))
                if savetest:
                    np.savetxt(Savepath+"area{0}test.txt".format(areaid), dataarray)
                else:
                    np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)
        time.sleep(0.001) # 延时0.1秒，免得CPU出问题

if __name__ == '__main__':
    # 0 對應額頭 ,1 對應下頜,2對應面部,3對應的是眼周,4不在人臉上
    area = 3
    savepath = "/home/gaofei/PycharmProjects/ElectroMagnetArea/SoarFacedata/"
    # test = True
    test = False
    main(area,test,savepath)
