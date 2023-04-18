from os import read
import queue
from codetiming import Timer
import asyncio
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
from itertools import count
import time
from matplotlib.animation import FuncAnimation
from numpy.core.numeric import True_
import matplotlib
import queue
import asyncio
import struct
import os
import sys
import time
import datetime
import atexit
import time
import numpy as np
from bleak import BleakClient
import matplotlib.pyplot as plt
from bleak import exc
import pandas as pd
import atexit
from multiprocessing import Pool
import multiprocessing

from src.solver import Solver_jac, Solver
from src.filter import Magnet_KF, Magnet_UKF
from src.preprocess import Calibrate_Data
from config import pSensor_smt, pSensor_large_smt, pSensor_small_smt, pSensor_median_smt, pSensor_imu
import cppsolver as cs
import cv2

'''The parameter user should change accordingly'''
# Change pSensor if a different sensor layout is used

# 采用的硬件型号的各个器件的空间位置
# pSensor = pSensor_large_smt
pSensor = pSensor_small_smt

# Change this parameter for different initial value for 1 magnet
# Change this parameter for different initial value for 1 magnet
#初始化参数方便线性优化找到最佳值
# params = np.array([40 / np.sqrt(2) * 1e-6, 40 / np.sqrt(2) * 1e-6,
#                    0, np.log(3), 1e-2 * (-2), 1e-2 * (2), 1e-2 * (11), 0, 0])
params = np.array([1e-2* (50), 1e-2* (50), 1e-2* (50), np.log(6), 1e-2 * (-2), 1e-2 * (2), 1e-2 * (11), 0, 0])
# Change this parameter for different initial value for 2 magnets
params2 = np.array([40/np.sqrt(2) * 1e-6, 40/np.sqrt(2) * 1e-6, 0, np.log(3) , 1e-2 * 6, 1e-2 * 0, 1e-2*(-1), 0, 0, 1e-2 * 5, 1e-2 * (4), 1e-2 * (-1), 0, 0,])

# Your adafruit nrd52832 ble address
#蓝牙模块的地址
# ble_address = "2A59A2D4-BCD8-4AF7-B750-E51195C1CA13"
# larger
# ble_address = "D2:CB:52:64:FB:60"
# small
ble_address = "C9:F1:B7:1C:20:F8"
# Absolute or relative path to the calibration data, stored in CSV
#标定补偿后的数据参数
cali_path = '/home/gaofei/magx/data/file_name.csv'


'''The calculation and visualization process'''
t = 0
matplotlib.use('Qt5Agg')
#Uart串口的地址
# Nordic NUS characteristic for RX, which should be writable
UART_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
# Nordic NUS characteristic for TX, which should be readable
UART_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
result = []
worklist = multiprocessing.Manager().Queue()


results = multiprocessing.Manager().Queue()
results2 = multiprocessing.Manager().Queue()
# 1 是左上 2是右上 3是左边下 4 是右下
areaid = multiprocessing.Manager().Queue()

currentx =10
currenty =10
currentz =10
def format_coord(x, y):
    global currentx
    global currenty
    global currentz
    return 'x坐标为%1.4f, y坐标为%1.4f, y坐标为%1.4f' % (currentx, currenty ,currentz)


def text(self, x, y, z, s, zdir=None, **kwargs):
    """
    Add text to the plot. kwargs will be passed on to Axes.text,
    except for the *zdir* keyword, which sets the direction to be
    used as the z direction.
    """
    global currentx
    global currenty
    global currentz
    return 'x坐标为%1.4f, y坐标为%1.4f, y坐标为%1.4f' % (currentx, currenty, currentz)
def end():
    print('End of the program')
    sys.exit(0)


def calculation_parallel(magcount=1, use_kf=0, use_wrist=False):
    global worklist
    global params
    global params2
    global results
    global results2
    global pSensor
    global areaid

    myparams1 = params
    myparams2 = params2
    while True:
        # 添加结束按键
        # k = cv2.waitKey() & 0xFF
        # if k == ord("k"):
        #     break
        if not worklist.empty():
            # 获取磁感计的数据
            datai = worklist.get()
            datai = datai.reshape(-1, 3)
            # print("real time datea :")
            # print(datai)
            # resulti [gx, gy, gz, m, x0,y0,z0, theta0, phy0, x1, y1, z1, theta1, phy1]
            if magcount == 1:
                if np.max(np.abs(myparams1[4:7])) > 1:
                    myparams1 = params
                # 非线性优化求解得到空间位置
                resulti = cs.solve_1mag(
                    datai.reshape(-1), pSensor.reshape(-1), myparams1)
                myparams1 = resulti

                # 此处查看m是否是定值
                # print("the constant value of m is :" ,myparams1[3])
                result = [resulti[4] * 1e2,
                          resulti[5] * 1e2, resulti[6] * 1e2]
                results.put(result)
                # print("Position: {:.2f}, {:.2f}, {:.2f}, dis={:.2f}".format(
                #     result[0],
                #     result[1],
                #     result[2],
                #     np.sqrt(
                #         result[0] ** 2 + result[1] ** 2 + result[2] ** 2)))
            #     计算当前位置处在人脸的上班部分还是下班部分
            #     if result[2]>-10 and result[0]>9 and result[0]<19 and result[1] >-12 and result[1]<7:
            #         print("it is up face ")
            #     elif result[2]<-12 and result[0]>9 and result[0]<19 and result[1] >-12 and result[1]<7:
            #         print("it is down face")
            #     else:
            #         print("no touch")

                # if result[0]>2 and result[0]>9 and result[0]<19 and result[1] >-12 and result[1]<7:
                #     print("it is up face ")
                # elif result[0]<-2 and result[0]>9 and result[0]<19 and result[1] >-12 and result[1]<7:
                #     print("it is down face")
                # else:
                #     print("no touch")

                # if result[0] > 2 and result[0]<18 and result[1] >1 and result[1] <8 and result[2]<6:
                #     areaid.put(1)
                #     print("当前在左脸上半部分 ")
                # elif result[0] < 0 and result[0] >-11 and result[1] >1 and result[1]<8 and result[2]<6:
                #     areaid.put(3)
                #     print("当前在左脸下半部分")
                # elif result[0]>2 and result[0]<18 and result[1]>-10 and result[1]<0 and result[2]<6:
                #     areaid.put(2)
                #     print("当前在右脸上半部分")
                # elif result[0]>-11 and result[0]<0 and result[1]>-10 and result[1]<0 and result[2]<6:
                #     areaid.put(4)
                #     print("当前在右脸下半部分")
                # else:
                #     areaid.put(0)
                #     print("无接触")
                # print(result)
                if ((result[0]-14)**2 +(result[1]+0.3)**2 +(result[2]-5.6)**2<50 or (result[0]-14)**2 +(result[1]-7.2)**2 +(result[2]+2.6)**2<25 or (result[0]-13)**2 +(result[1]+5.8)**2 +(result[2]-3.8)**2<25) and result[2]<8:
                    areaid.put(1)
                    print("当前在额头区域 ")
                elif (result[0]-2)**2 +(result[1]-7)**2 +(result[2]-6.8)**2<6:
                    areaid.put(2)
                    print("靠近左眼，危险")
                elif (result[0]-5.6)**2 +(result[1]+5.8)**2 +(result[2]-6.5)**2<6:
                    areaid.put(3)
                    print("靠近右眼，危险")
                elif (result[0]+0.98)**2 +(result[1]-6.48)**2 +(result[2]-4.8)**2<24 or (result[0]+5.50)**2 +(result[1]-8.50)**2 +(result[2]+2.96)**2<24:
                    areaid.put(4)
                    print("当前在左脸颊区域")
                elif (result[0]+1.5)**2 +(result[1]+8.2)**2 +(result[2]-5.2)**2<24 or (result[0]+4.96)**2 +(result[1]+9.88)**2 +(result[2]-3.2)**2<24:
                    areaid.put(5)
                    print("当前在右脸颊区域")
                else:
                    areaid.put(0)
                    print("无接触")
            elif magcount == 2:
                if np.max(
                        np.abs(myparams2[4: 7])) > 1 or np.max(
                        np.abs(myparams2[9: 12])) > 1:
                    myparams2 = params2

                resulti = cs.solve_2mag(
                    datai.reshape(-1), pSensor.reshape(-1), myparams2)
                myparams2 = resulti
                result = [resulti[4] * 1e2,
                          resulti[5] * 1e2, resulti[6] * 1e2]
                results.put(result)
                result2 = [resulti[9] * 1e2,
                           resulti[10] * 1e2, resulti[11] * 1e2]
                results2.put(result2)
                print(
                    "Mag 1 Position: {:.2f}, {:.2f}, {:.2f}, dis={:.2f} \n Mag 2 Position: {:.2f}, {:.2f}, {:.2f}, dis={:.2f}". format(
                        result[0],
                        result[1],
                        result[2],
                        np.sqrt(
                            result[0] ** 2 +
                            result[1] ** 2 +
                            result[2] ** 2),
                        result2[0],
                        result2[1],
                        result2[2],
                        np.sqrt(
                            result2[0] ** 2 +
                            result2[1] ** 2 +
                            result2[2] ** 2)))


async def task(name, work_queue):
    timer = Timer(text=f"Task {name} elapsed time: {{: .1f}}")
    while not work_queue.empty():
        delay = await work_queue.get()
        print(f"Task {name} running")
        timer.start()
        await asyncio.sleep(delay)
        timer.stop()

async def show_mag(magcount=1):
    global t
    global pSensor
    global results
    global results2
    global currentx
    global currenty
    global currentz
    myresults = np.array([[0, 10, 10]])
    myresults2 = np.array([[0, 0, 10]])
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(fc='whitesmoke',
                 projection='3d'
                 )
    ax.view_init(elev=10,  # 仰角
                 azim=20 # 方位角
                 )


    # 画出xyz坐标轴
    ax.quiver(0, 0, 0, 0, 0, 8,color="b")
    # 画起点为(0,0,0),终点为(0,1,0)的向量
    ax.quiver(0, 0, 0, 0, 8, 0,color="g")
    # 画起点为(0,0,0),终点为(1,0,0)的向量
    ax.quiver(0, 0, 0, 8, 0, 0,color="r")
    x = [1, 0, 0]
    y = [0, 1, 0]
    z = [0, 0, 1]
    X,Y,Z = zip(x, y, z)
    axis = zip(X,Y,Z)
    for pos in list('xyz'.upper()):
        ax.text(*eval(pos),
                s=pos,
                fontsize=6,
                color='darkgreen')

    # 画出上下两个区域的方框
    # 上面区域的代码
    # x = [11.5618, 13.8602, -7.9714, -11.0521, -9.4149,14.5238 , -9.6392, -9.6309]
    # y = [8.5126  ,11.0381,  8.9825, 10.9664, -17.3226, -9.6304, -10.9245,-10.4244]
    # z = [-11.1332, -6.6660, 10.9203, -0.3360, -9.5996, 4.0517, 7.6999, -0.2758]
    # A, B, C, D, E, F, G, H = zip(x, y, z)
    # #
    # # # 绘制 3D 图形
    # lines_1 = zip(A, B, C, D, A, E, F, G, C, B, F)
    # ax.plot3D(*lines_1,
    #           zdir='z',  #
    #           c='k',  # color
    #           marker='o',  # 标记点符号
    #           mfc='r',  # marker facecolor
    #           mec='g',  # marker edgecolor
    #           ms=10,  # size
    #           )
    #
    # lines_2 = zip(D, H, E, G, H)
    # ax.plot(*lines_2,
    #         zdir='z',  #
    #         c='k',  # color
    #         marker='o',  # 标记点符号
    #         mfc='r',  # marker facecolor
    #         mec='g',  # marker edgecolor
    #         ms=10,  # size
    #         )


    # TODO: add title
    ax.set_xlabel('x(cm)')
    ax.set_ylabel('y(cm)')
    ax.set_zlabel('z(cm)')
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    ax.set_zlim([-50, 50])
    Xs = 1e2 * pSensor[:, 0]
    Ys = 1e2 * pSensor[:, 1]
    Zs = 1e2 * pSensor[:, 2]

    XXs = Xs
    YYs = Ys
    ZZs = Zs
    colorlist =["r","b","g","c","m","y","cyan","purple"]
    sensorIdlist= ["Sensor1","Sensor2","Sensor3","Sensor4","Sensor5","Sensor6","Sensor7","Sensor8"]
    for i in range(8):
        ax.scatter(XXs[i], YYs[i], ZZs[i], c=colorlist[i], s=10, alpha=0.9,label=sensorIdlist[i])
    # ax.scatter(XXs, YYs, ZZs, c='r', s=5, alpha=0.5)
    ax.legend(loc='best')

    (magnet_pos,) = ax.plot(t / 100.0 * 5, t / 100.0 * 5, t /
                            100.0 * 5, linewidth=3, animated=True)
    if magcount == 2:
        (magnet_pos2,) = ax.plot(t / 100.0 * 5, t / 100.0 * 5, t /
                                 100.0 * 5, linewidth=3, animated=True)

    # zdirs = ('position')
    # label = '(%d, %d, %d), dir=%s' % (magnet_pos[0], magnet_pos[1], magnet_pos[2], zdirs)
    # ax.text(magnet_pos[0], magnet_pos[1], magnet_pos[2], label, zdirs)



    plt.show(block=False)
    plt.pause(0.1)
    bg = fig.canvas.copy_from_bbox(fig.bbox)
    # ax.text()
    ax.draw_artist(magnet_pos)
    fig.canvas.blit(fig.bbox)
    # timer = Timer(text=f"frame elapsed time: {{: .5f}}")

    # 画出图片
    # img = Image.open('/home/gaofei/magx/data/4lv9c8ee.png')
    # plt.figure("Image")  # 图像窗口名称
    # plt.imshow(img)

    # 画出磁铁的实时位置
    while True:
        # 添加结束按键
        # k = cv2.waitKey() & 0xFF
        # if k == ord("k"):
        #     break
        # timer.start()
        fig.canvas.restore_region(bg)
        # update the artist, neither the canvas state nor the screen have
        # changed

        # update myresults
        # 此处可以添加卡尔曼滤波
        if not results.empty():
            myresult = results.get()
            myresults = np.concatenate(
                [myresults, np.array(myresult).reshape(1, -1)])

        if myresults.shape[0] > 30:
            myresults = myresults[-30:]

        x = myresults[:, 0]
        y = myresults[:, 1]
        z = myresults[:, 2]

        xx = x
        yy = y
        zz = z

        magnet_pos.set_xdata(xx)
        magnet_pos.set_ydata(yy)
        magnet_pos.set_3d_properties(zz, zdir='z')
        ax.draw_artist(magnet_pos)
        # 添加实时点坐标显示
        currentx=xx[-1]
        currenty=yy[-1]
        currentz=zz[-1]

        # ax.format_coord = format_coord
        # ax.button_pressed =1
        ax.format_coord=format_coord



        if magcount == 2:
            if not results2.empty():
                myresult2 = results2.get()
                myresults2 = np.concatenate(
                    [myresults2, np.array(myresult2).reshape(1, -1)])

            if myresults2.shape[0] > 30:
                myresults2 = myresults2[-30:]
            x = myresults2[:, 0]
            y = myresults2[:, 1]
            z = myresults2[:, 2]

            xx = x
            yy = y
            zz = z
            # 添加点的实时坐标
            magnet_pos2.set_xdata(xx)
            magnet_pos2.set_ydata(yy)
            magnet_pos2.set_3d_properties(zz, zdir='z')
            ax.draw_artist(magnet_pos2)


        # copy the image to the GUI state, but screen might not changed yet
        # 注释掉此行方便各个角度观看
        fig.canvas.blit(fig.bbox)
        # flush any pending GUI events, re-painting the screen if needed
        fig.canvas.flush_events()
        await asyncio.sleep(0)

        # timer.stop()


def notification_handler(sender, data):
    """Simple notification handler which prints the data received."""
    global pSensor
    global worklist
    num = int(pSensor.size/3)

    all_data = []
    sensors = np.zeros((num, 3))
    current = [datetime.datetime.now()]
    calibration = np.load('result/calibration.npz')
    offset = calibration['offset'].reshape(-1)
    # print("offset")
    # print(offset)
    scale = calibration['scale'].reshape(-1)
    # print("scale")
    # print(scale)
    for i in range(num):
        sensors[i, 0] = struct.unpack('f', data[12 * i: 12 * i + 4])[0]
        sensors[i, 1] = struct.unpack('f', data[12 * i + 4: 12 * i + 8])[0]
        sensors[i, 2] = struct.unpack('f', data[12 * i + 8: 12 * i + 12])[0]
        # print("Sensor " + str(i+1)+": "+str(sensors[i, 0]) + ", " + str(sensors[i, 1]) + ", " + str(sensors[i, 2]))
        current.append(
            "(" + str(sensors[i, 0]) + ", " + str(sensors[i, 1]) + ", " +
            str(sensors[i, 2]) + ")")
        # battery_voltage = struct.unpack('f', data[12 * num: 12 * num + 4])[0]
        # print("Battery voltage: " + str(battery_voltage))
    sensors = sensors.reshape(-1)
    sensors = (sensors - offset) / scale * np.mean(scale)

    if len(all_data) > 3:
        print("hello we come here")
        sensors = (sensors + all_data[-1] + all_data[-2]) / 3
    all_data.append(sensors)
    worklist.put(sensors)
    # print("############")



async def run_ble(address, loop):
    async with BleakClient(address, loop=loop) as client:
        # wait for BLE client to be connected
        x = await client.is_connected()
        print("Connected: {0}".format(x))
        print("Press Enter to quit...")
        # wait for data to be sent from client
        await client.start_notify(UART_TX_UUID, notification_handler)
        while True:
            await asyncio.sleep(0.01)
            # data = await client.read_gatt_char(UART_TX_UUID)

def showimag():
    global areaid
    sourceimg = cv2.imread("/home/gaofei/magx/data/4lv9c8ee.png")
    mask1 = cv2.imread("/home/gaofei/magx/data/mask1.png")
    mask2 = cv2.imread("/home/gaofei/magx/data/mask2.png")
    mask3 = cv2.imread("/home/gaofei/magx/data/mask3.png")
    mask4 = cv2.imread("/home/gaofei/magx/data/mask4.png")
    mask5 = cv2.imread("/home/gaofei/magx/data/mask5.png")
    myresults = np.array([[0]])
    while True:
        if not areaid.empty():
            myresult = areaid.get()
            myresults = np.concatenate(
                [myresults, np.array(myresult).reshape(1, -1)])
        if myresults.shape[0] > 0:
            myresults = myresults[-1:]
        if myresults[0][0]==0:
            # print("current is" ,areaid)
            cv2.imshow('image', sourceimg)
            cv2.waitKey(30)
        elif myresults[0][0]==1:
            # print("current is" ,areaid)
            cv2.imshow('image', mask1)
            cv2.waitKey(30)
        elif myresults[0][0]==2:
            # print("current is" ,areaid)
            cv2.imshow('image', mask2)
            cv2.waitKey(30)
        elif myresults[0][0]==3:
            # print("current is" ,areaid)
            cv2.imshow('image', mask3)
            cv2.waitKey(30)
        elif myresults[0][0]==4:
            # print("current is" ,areaid)
            cv2.imshow('image', mask4)
            cv2.waitKey(30)
        elif myresults[0][0]==5:
            # print("current is" ,areaid)
            cv2.imshow('image', mask5)
            cv2.waitKey(30)
        else:
            cv2.imshow('image', sourceimg)
            cv2.waitKey(30)
    cv2.destroyAllWindows()

        # print("currentid is ",myresults[0])



async def main(magcount=1):
    """
    This is the main entry point for the program
    """
    # Address of the BLE device
    global ble_address
    global threadinglist
    address = (ble_address)

    # Run the tasks
    with Timer(text="\nTotal elapsed time: {:.1f}"):
        # 添加一个显示图片的线程
        # showimg = multiprocessing.Process(target=showimag)
        # showimg.start()
        getposition=multiprocessing.Process(
            target=calculation_parallel, args=(magcount, 1, False))
        getposition.start()
        await  asyncio.gather(
            asyncio.create_task(run_ble(address, asyncio.get_event_loop())),
            asyncio.create_task(show_mag(magcount))
        )



if __name__ == '__main__':
    # if True:
    #     calibration = Calibrate_Data(cali_path)
    #     [offset, scale] = calibration.cali_result()
    #     if not os.path.exists('result'):
    #         os.makedirs('result')
    #     np.savez('result/calibration.npz', offset=offset, scale=scale)
    #     print("hello: ",np.mean(scale))
        # calibration.show_cali_result()
    asyncio.run(main(1))  # For tracking 1 magnet

    # asyncio.run(main(2)) # For tracking 2 magnet
