import threading
import time
import serial
import numpy as np

sensorzero= np.zeros((1,9))
realtimebuffer = [sensorzero,sensorzero]
datasaver =[]

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
            realtimebuffer[0] = sensorzero
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
            realtimebuffer[1] = sensorzero

def chosendata():
    chosendataresult = np.zeros((1, 9))
    while True:
        # print("realtimeleft",realtimebuffer[0])
        # print("realtimeright",realtimebuffer[1])
        if realtimebuffer[0].any() !=0 and realtimebuffer[1].any() !=0:
            # leftsum = np.sum(realtimebuffer[0]*realtimebuffer[0])
            # rightsum = np.sum(realtimebuffer[1]*realtimebuffer[1])
            leftsum =abs(realtimebuffer[0][0][0])
            rightsum = abs(realtimebuffer[1][0][0])
            if leftsum > rightsum:
                chosendataresult = realtimebuffer[0]
                datasaver.append(chosendataresult[0][:9])
                print("chose one left")
        #         print(chosendataresult)
            else:
                chosendataresult = realtimebuffer[1]
                datasaver.append(chosendataresult[0][:9])
                print("chose one right")
        #         print(chosendataresult)
        # elif (realtimebuffer[0] is None) and (realtimebuffer[1] is not None):
        elif realtimebuffer[0].any() ==0 and realtimebuffer[1].any() !=0:
            chosendataresult = realtimebuffer[1]
            datasaver.append(chosendataresult[0][:9])
            print("chose 2 right")
            # print(chosendataresult)
        # elif (realtimebuffer[0] is not None) and (realtimebuffer[1] is None):
        elif realtimebuffer[0].any() !=0 and realtimebuffer[1].any() ==0:
            chosendataresult = realtimebuffer[0]
            datasaver.append(chosendataresult[0][:9])
            print("chose 2 left")
            # print(chosendataresult)
        # np.savetxt("./sensordataright.txt", np.array(datasaver).reshape((-1, 9)))
        # print(chosendataresult)
        # time.sleep(0.033)
        # time.sleep(0.033)



if __name__ == '__main__':
    left_thread = threading.Thread(target=serialleft)
    right_thread = threading.Thread(target=serialrifht)
    chosen_thread = threading.Thread(target=chosendata)

    # 开启线程
    left_thread.start()
    right_thread.start()
    chosen_thread.start()
