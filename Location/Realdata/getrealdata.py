import serial
import time
import numpy as np
import matplotlib.pyplot as plt

def savedata():
    # 讀取實時數據
    ser = serial.Serial("/dev/ttyUSB0",256000,timeout = 0.01)
    ser.flushInput()
    datasaver = []
    while True:
        currentdata = ser.readline()
        if currentdata !=b'' and currentdata !=b'\n':
            # starttime= time.time()
            currentdata = str(currentdata, 'UTF-8')
            currentdatalist = currentdata.split('\r\n')[0]
            currentdatalist = currentdatalist.split(",")
            dataarray = np.array(currentdatalist,dtype='float16').reshape((-1,18))
            print("data",dataarray[0][:9])
            datasaver.append(dataarray[0][:9])
            np.savetxt("./sensordata.txt",np.array(datasaver).reshape((-1,9)))
            # print("data",datasaver)
            # endtime = time.time()
        time.sleep(0.001) # 延时0.1秒，免得CPU出问题



def showdata():
    all_data = np.loadtxt("./sensordata.txt")
    mag_datacoil1 = all_data[:,:3]
    mag_datacoil2 = all_data[:,3:6]
    mag_datacoil3 = all_data[:,6:9]
    # print(mag_data.shape)

    # fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    calibratex = mag_datacoil1[:, 0]
    calibratey = mag_datacoil1[:, 1]
    calibratez = mag_datacoil1[:, 2]
    ax.scatter3D(calibratex, calibratey, calibratez, color="red")

    calibratex1 = mag_datacoil2[:, 0]
    calibratey1 = mag_datacoil2[:, 1]
    calibratez1 = mag_datacoil2[:, 2]
    ax.scatter3D(calibratex1, calibratey1, calibratez1, color="green")

    calibratex2 = mag_datacoil3[:, 0]
    calibratey2 = mag_datacoil3[:, 1]
    calibratez2 = mag_datacoil3[:, 2]
    ax.scatter3D(calibratex2, calibratey2, calibratez2, color="blue")
    plt.title("3D scatter plot")
    plt.show()

if __name__ =="__main__":
    showdata()
    # savedata()
