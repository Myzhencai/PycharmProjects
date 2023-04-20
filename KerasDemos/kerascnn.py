#實現keras庫下的cnn分類器
# 最後寫cnn以免浪費時間
# 實現基於keras實現雙向lstm
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import cv2
import serial
import time

def loaddata(filepath):
    all_data = np.loadtxt(filepath)
    return all_data

# ------------------------------定义参数------------------------------
TIME_STEPS = 1  # 时间点数据 每次读取1行共28次 same as the height of the image
INPUT_SIZE = 9  # 每行读取28个像素点 same as the width of the image
BATCH_SIZE = 128 # 每个批次训练50张图片
BATCH_INDEX = 0
OUTPUT_SIZE = 4  # 每张图片输出分类矩阵
CELL_SIZE = 128  # RNN中隐藏单元
LR = 0.001  # 学习率

# ------------------------------數據加載------------------------------
path = "/home/gaofei/PycharmProjects/ElectroMagnetArea/SoarFacedata/megedData.txt"
# path = "/home/gaofei/PycharmProjects/ElectroMagnetArea/SoarFacedata7new/megedData.txt"
dataSet = loaddata(path)
x = dataSet[:,:9]
y = dataSet[:,9:13]

X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.25)
X_train = X_train.reshape((X_train.shape[0],1,X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0],1,X_test.shape[1]))

# ------------------------------創建模型------------------------------
# 參數設置
model = Sequential()
model.add(Bidirectional(LSTM(96, return_sequences=True), input_shape=(TIME_STEPS, INPUT_SIZE)))
model.add(Bidirectional(LSTM(56, return_sequences=True), input_shape=(TIME_STEPS, 192)))
model.add(TimeDistributed(Dense(4, activation='sigmoid')))

# ------------------------------確定優化器------------------------------
adam = Adam(LR)
model.compile(optimizer=adam,  # 加速神经网络
              loss='categorical_crossentropy',  # 损失函数
              metrics=['accuracy'])  # 计算误差或准确率
print(model.summary())
# --------------------------------训练和预测------------------------------
for step in range(4001):
    # 分批截取数据 BATCH_INDEX初始值为0 BATCH_SIZE为50 取28个步长和28个INPUT_SIZE
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :, :]
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :].reshape((-1,1,4))
    # 计算误差
    cost = model.train_on_batch(X_batch, Y_batch)

    # 累加参数
    BATCH_INDEX += BATCH_SIZE
    # 如果BATCH_INDEX累加大于总体的个数 则重新赋值0开始分批计算
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    # 每隔200步输出
    if step % 200 == 0:
        # 评价算法
        y_test = y_test.reshape((-1,1,4))
        cost, accuracy = model.evaluate(
            X_test, y_test,
            batch_size=y_test.shape[0],
            verbose=1)
        print('test cost: ', cost, 'test accuracy: ', accuracy)


lefteye = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/lefteye.png")
leftface = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/leftface.png")
lefthead = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/lefthead.png")
leftjaw = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/leftjaw.png")

ser = serial.Serial("/dev/ttyUSB0",256000,timeout = 0.01) # 开启com3口，波特率115200，超时5
ser.flushInput() # 清空缓冲区
while True:
    currentdata = ser.readline() # 获取串口缓冲区数据
    if currentdata !=b'' :
        starttime= time.time()
        currentdata = str(currentdata, 'UTF-8')
        currentdatalist = currentdata.split('\r\n')[0]
        currentdatalist = currentdatalist.split(",")
        dataarray = np.array(currentdatalist,dtype='float32').reshape((-1,9))
        input = dataarray.reshape((1, 1, 9))
        result = model.predict(input)
        result = result[0][0]
        pred_y = np.argmax(result, 0)
        endtime = time.time()

        print("單次運算的時間:",endtime-starttime)
        if pred_y==0:
            print("當前在額頭區域")
            cv2.imshow('image', lefthead)
            cv2.waitKey(30)
        elif pred_y==1:
            # print("current is" ,areaid)
            print("當前在下頜線區域")
            cv2.imshow('image', leftjaw)
            cv2.waitKey(30)
        elif pred_y==2:
            # print("current is" ,areaid)
            print("當前在面部區域")
            cv2.imshow('image', leftface)
            cv2.waitKey(30)
        elif pred_y==3:
            # print("current is" ,areaid)
            print("當前在眼周區域")
            cv2.imshow('image', lefteye)
            cv2.waitKey(30)

input = X_test[0].reshape((1,1,9))
result = model.predict(input)
pred_y = np.argmax(result, 1)

# 保存模型
model.save('KerasBiLstm.h5')
newmodel = keras.models.load_model('KerasBiLstm.h5')
print(newmodel.summary())
