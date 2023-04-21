#實現keras庫下的mobilenet分類器
# https://github.com/xiaochus/MobileNetV3/blob/master/train_cls.py
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
import numpy as np
from sklearn.model_selection import train_test_split
from mobilenet.model.mobilenet_v3_small import MobileNetV3_Small
from keras.optimizers import Adam
import keras
# 效果一般只有0.8的準確率

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
# model = Sequential()
# model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(TIME_STEPS, INPUT_SIZE)))
# # model.add(Bidirectional(LSTM(56, return_sequences=True), input_shape=(TIME_STEPS, 192)))
# model.add(TimeDistributed(Dense(4, activation='sigmoid')))
input_shape =(1,1,9)
model = MobileNetV3_Small(input_shape, 4).build()

# ------------------------------確定優化器------------------------------
adam = Adam(LR)
model.compile(optimizer=adam,  # 加速神经网络
              loss='categorical_crossentropy',  # 损失函数
              metrics=['accuracy'])  # 计算误差或准确率
print(model.summary())

# --------------------------------训练和预测------------------------------
for step in range(10000):
    # 分批截取数据 BATCH_INDEX初始值为0 BATCH_SIZE为50 取28个步长和28个INPUT_SIZE
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :, :].reshape((-1,1,1,9))
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :]
    # 计算误差
    cost = model.train_on_batch(X_batch, Y_batch)

    # 累加参数
    BATCH_INDEX += BATCH_SIZE
    # 如果BATCH_INDEX累加大于总体的个数 则重新赋值0开始分批计算
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    # 每隔200步输出
    if step % 200 == 0:
        # 评价算法
        y_test = y_test
        X_test = X_test.reshape((-1,1,1,9))
        cost, accuracy = model.evaluate(
            X_test, y_test,
            batch_size=y_test.shape[0],
            verbose=1)
        print('test cost: ', cost, 'test accuracy: ', accuracy)
