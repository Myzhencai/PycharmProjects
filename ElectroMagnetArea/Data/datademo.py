# import serial # 导入串口包
# import time # 导入时间包
#
# ser = serial.Serial("/dev/ttyUSB0",256000,timeout = 0.01) # 开启com3口，波特率115200，超时5
# ser.flushInput() # 清空缓冲区
#
# def main():
#     while True:
#         currentdata = ser.readline() # 获取串口缓冲区数据
#         if currentdata !=b'' :
#             currentdata = str(currentdata, 'UTF-8')
#             currentdatalist = currentdata.split('\r\n')[0]
#             print("currentdata",currentdatalist)
#         else:
#             print("current have no data ")
#         time.sleep(0.001) # 延时0.1秒，免得CPU出问题
#
# if __name__ == '__main__':
#     # 0 對應額頭 ,1 對應下頜,2對應面部,3對應的是眼周,4不在人臉上
#     main()


import numpy as np

field_adj = np.array([[1,2,3],[4,5,6],[7,8,9]])

doublefeild  = field_adj ** 2

line0 =field_adj[:, 0]
line1 =field_adj[:, 1]
line2 =field_adj[:, 2]

testfeild = np.array([field_adj[:, 1]*field_adj[:, 2],
    field_adj[:, 0]*field_adj[:, 2],
    field_adj[:, 0]*field_adj[:, 1]])


features_1 = np.hstack((field_adj ** 2,
    np.array([field_adj[:, 1]*field_adj[:, 2],
    field_adj[:, 0]*field_adj[:, 2],
    field_adj[:, 0]*field_adj[:, 1]]).T, np.ones((field_adj.shape[0], 1))))
