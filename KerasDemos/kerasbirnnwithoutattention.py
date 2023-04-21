# https://blog.paperspace.com/bidirectional-rnn-keras/
# https://vimsky.com/examples/usage/python-tf.keras.layers.Embedding-tf.html
# 實現birnn效果非常好0.98

import numpy as np
import keras
from sklearn.model_selection import train_test_split
import cv2
import serial # 导入串口包
import time

# ------------------------------定义参数------------------------------
TIME_STEPS = 1  # 时间点数据 每次读取1行共28次 same as the height of the image
INPUT_SIZE = 9  # 每行读取28个像素点 same as the width of the image
BATCH_SIZE = 128 # 每个批次训练50张图片
BATCH_INDEX = 0
OUTPUT_SIZE = 4  # 每张图片输出分类矩阵
CELL_SIZE = 128  # RNN中隐藏单元
LR = 0.001  # 学习率

# ---------------------------载入数据及预处理---------------------------
def loaddata(filepath):
    all_data = np.loadtxt(filepath)
    return all_data

path = "/home/gaofei/PycharmProjects/ElectroMagnetArea/SoarFacedata/megedData.txt"
dataSet = loaddata(path)
x = dataSet[:,:9]
y = dataSet[:,9:13]

# 區分訓練集合和驗證集合
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.25)
X_train = X_train.reshape((X_train.shape[0],1,X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0],1,X_test.shape[1]))

# ---------------------------创建RNN神经网络---------------------------
# 创建RNN模型
model = keras.Sequential()
# 第一個數是訓練數據的最大值
model.add(keras.layers.Embedding(60000, 18, input_length = 9))
model.add(keras.layers.SpatialDropout1D(0.4))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(20, dropout=0.05, recurrent_dropout=0.2)))
model.add(keras.layers.Dense(4, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
model.summary()

# --------------------------------训练和预测------------------------------
cost_list = []
acc_list = []
step_list = []
for step in range(8000):
    # 分批截取数据 BATCH_INDEX初始值为0 BATCH_SIZE为50 取28个步长和28个INPUT_SIZE
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :, :].reshape((-1,9))
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
        X_test = X_test.reshape((-1,9))
        cost, accuracy = model.evaluate(
            X_test, y_test,
            batch_size=y_test.shape[0],
            verbose=False)
        # 写入列表
        cost_list.append(cost)
        acc_list.append(accuracy)
        step_list.append(step)
        print('test cost: ', cost, 'test accuracy: ', accuracy)
model.save('KerasBirnn.h5')
newmodel = keras.models.load_model('KerasBirnn.h5')
print(newmodel.summary())