import keras
import numpy as np
import tensorflow as tf
np.random.seed(1337)
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.layers import LSTM
# https://blog.csdn.net/cunzai1985/article/details/108751972?ydreferer=aHR0cHM6Ly93d3cuYmFpZHUuY29tL2xpbms%2FdXJsPU04S21tUFlpMDMxS2EzcFJ5UEpNVDdJSUxMOGd0anZKSkpHcXpkZHp5ZVFJMU5oWmxaT3dUdFAyRFFpR0s3NllqOXBPemlnQWxnMm9ZVjJ0MndEX09VMXRJNERuLTFVeXJ1ZGlMeTFCRlltJndkPSZlcWlkPTliZmM4NDZhMDAwMTIwMDcwMDAwMDAwNjY0M2QxOWJi

TIME_STEPS = 1  #等于图片高度(矩阵的行数)，每次读取一行，图片大小是28*28，因此需要读取28次
INPUT_SIZE = 9  #等于图片宽度（矩阵的高数），每次读取一行中的多少个像素，一行有28个
BATCH_SIZE = 128  #每次训练50张图片
BATCH_INDEX = 0  #生成数据
OUTPUT_SIZE = 4  #输出的尺寸，每次输出是0-9的数据，所以为10个[0,1,0 0 0 0 0 0 0 0]这个形式
CELL_SIZE = 128  #RNN里面的隐藏层个数
LR = 0.01  #学习率

def loaddata(filepath):
    all_data = np.loadtxt(filepath)
    return all_data
#数据
# (x_train, y_train), (x_test, y_test) = mnist.load_data() #从库中导入数据

# x_train = x_train.reshape(-1, 28, 28) / 255  #x数据需要归一化，不然数据太乱
# x_test = x_test.reshape(-1, 28, 28) / 255
# y_train = np_utils.to_categorical(y_train, 10) #y设成one hot数据标签
# y_test = np_utils.to_categorical(y_test, 10)

path = "/home/gaofei/PycharmProjects/ElectroMagnetArea/Data/HalfFace7/megedData.txt"
dataSet = loaddata(path)
x = dataSet[:,:9]
y = dataSet[:,9:13]
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.25)


#建立模型
model = Sequential()
model.add(SimpleRNN(  #直接调用封装的RNN层就可以了，就相当于已经完成了RNN模型的搭建
    units=CELL_SIZE,     #输出就是隐藏层个数
    input_shape=(TIME_STEPS, INPUT_SIZE), #输入尺寸就是28*28

))
model.add(Dense(OUTPUT_SIZE)) #最后接一个全连接层，就OK了，最后输出尺寸为10
model.add(Activation('softmax'))
#编译
adam = Adam(LR) #设置优化器参数
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=adam, loss='softmax_cross_entropy', metrics=['accuracy'])

#训练
for step in range(4001): #循环4001次
    #取x_train和y_train中的一部分数据
    x_batch = x_train[BATCH_INDEX:BATCH_SIZE+BATCH_INDEX, :]  #取出的尺寸为(50,28,28)
    x_batch = tf.reshape(x_batch, [-1, TIME_STEPS, INPUT_SIZE])
    y_batch = y_train[BATCH_INDEX:BATCH_SIZE+BATCH_INDEX, :]  #取出的尺寸为(50,10)

    cost = model.train_on_batch(x_batch, y_batch) #一次循环拿50个数据进行训练

    BATCH_INDEX += BATCH_SIZE #第一次去前50个数据，下一次就取50-100这50个数据，以此类推
    BATCH_INDEX = 0 if BATCH_INDEX >= x_train.shape[0] else BATCH_INDEX #如果数据取完了，就从头开始，没有就继续接着之前的取

#测试
    if step % 500 == 0:
        cost, accuracy = model.evaluate(x_test, y_test, batch_size=y_test.shape[0]) #一次性拿50个测试数据进行测试
        print(cost, accuracy)
