# https://blog.csdn.net/m0_37602827/article/details/104883494
# 基於keras 實現tcn，效果比較差
import numpy as np
from keras.models import Model
from keras.layers import add, Input, Conv1D, Activation, Flatten, Dense
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam


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

#------------------------------構建模型------------------------------
def ResBlock(x, filters, kernel_size, dilation_rate):
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(x)  # 第一卷积
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(r)  # 第二卷积
    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = Conv1D(filters, kernel_size, padding='same')(x)  # shortcut（捷径）
    o = add([r, shortcut])
    o = Activation('relu')(o)  # 激活函数
    return o

def TCN():
    # inputs = Input(shape=(28, 28))
    inputs = Input(shape=(TIME_STEPS, INPUT_SIZE))
    x = ResBlock(inputs, filters=32, kernel_size=3, dilation_rate=1)
    x = ResBlock(x, filters=32, kernel_size=3, dilation_rate=2)
    x = ResBlock(x, filters=16, kernel_size=3, dilation_rate=4)
    x = Flatten()(x)
    x = Dense(4, activation='softmax')(x)
    model = Model(input=inputs, output=x)

    # 查看网络结构
    adam = Adam(LR)
    # 编译模型
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # 返回模型
    return model
    # # 训练模型
    # model.fit(train_x, train_y, batch_size=500, nb_epoch=30, verbose=2, validation_data=(valid_x, valid_y))
    # # 评估模型
    # pre = model.evaluate(test_x, test_y, batch_size=500, verbose=2)
    # print('test_loss:', pre[0], '- test_acc:', pre[1])
TCNnet = TCN()
TCNnet.summary()

# # --------------------------------训练和预测------------------------------
for step in range(10000):
    # 分批截取数据 BATCH_INDEX初始值为0 BATCH_SIZE为50 取28个步长和28个INPUT_SIZE
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :, :]
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :]
    # 计算误差
    cost = TCNnet.train_on_batch(X_batch, Y_batch)

    # 累加参数
    BATCH_INDEX += BATCH_SIZE
    # 如果BATCH_INDEX累加大于总体的个数 则重新赋值0开始分批计算
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    # 每隔200步输出
    if step % 200 == 0:
        # 评价算法
        y_test = y_test
        cost, accuracy = TCNnet.evaluate(
            X_test, y_test,
            batch_size=y_test.shape[0],
            verbose=1)
        print('test cost: ', cost, 'test accuracy: ', accuracy)

# 因爲網絡太差勁了所以放棄此方法
