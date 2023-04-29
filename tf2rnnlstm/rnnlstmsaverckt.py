import tensorflow as tf
# import numpy as np
from sklearn.model_selection import train_test_split
# from tensorflow.python.framework import graph_util
# from tensorflow.python.platform import gfile
import numpy as np
import serial # 导入串口包
import time # 导入时间包
import cv2

# https://www.cnblogs.com/zerotoinfinity/p/10242849.html

# 加載數據
def loaddata(filepath):
    all_data = np.loadtxt(filepath)
    return all_data
#
# tf.set_random_seed(1)
# np.random.seed(1)
tf.compat.v1.set_random_seed(777)
tf.compat.v1.disable_eager_execution()

# Hyper Parameters
BATCH_SIZE =128
TIME_STEP = 1          # rnn time step / image height
INPUT_SIZE = 9         # rnn input size / image width
LR = 0.01               # learning rate
CLASS_NUM = 4
BasicLSTMCell_NUM = 128

# 加載數據
# path = "/home/gaofei/PycharmProjects/ElectroMagnetArea/Data/HalfFace7/megedData.txt"
path = "/home/gaofei/PycharmProjects/ElectroMagnetArea/SoarFacedata/megedData.txt"

# 實時顯示結果的部分
# lefteye = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/lefteye.png")
# leftface = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/leftface.png")
# lefthead = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/lefthead.png")
# leftjaw = cv2.imread("/home/gaofei/PycharmProjects/ElectroMagnetArea/fivearea/data/newarea/leftjaw.png")


dataSet = loaddata(path)
x = dataSet[:,:9]
y = dataSet[:,9:13]

# 區分訓練集合和驗證集合
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.25)


epochs_completed = 0
index_in_epoch = 0
num_examples = X_train.shape[0]
    # for splitting out batches of data
def next_batch(batch_size):
    global X_train
    global y_train
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        X_train = X_train[perm]
        y_train = y_train[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return X_train[start:end], y_train[start:end]
# pb_file_path = "/home/gaofei/PycharmProjects/ElectroMagnetArea/demodata/"

with tf.Session(graph=tf.Graph()) as sess:
    # 參數需要導出來的參數佔位符
    tf_x = tf.placeholder(tf.float32, [None, TIME_STEP * INPUT_SIZE],name='x')  # (128,9)     # shape(batch, 784)
    image = tf.reshape(tf_x, [-1, TIME_STEP, INPUT_SIZE])  # (128,1,9)                 # (batch, height, width, channel)
    tf_y = tf.placeholder(tf.int32, [None, CLASS_NUM],name='y')  # (128,4)                          # input y

    #神經網絡運算代碼
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=BasicLSTMCell_NUM)
    # (128,28,128)
    outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
        rnn_cell,  # cell you have chosen
        image,  # input
        initial_state=None,  # the initial hidden state
        dtype=tf.float32,  # must given if set initial_state = None
        time_major=False,  # False: (batch, time step, input); True: (time step, batch, input)
    )
    output = tf.layers.dense(outputs[:, -1, :], CLASS_NUM)#爲什麼呢不能添加名字不屬於運算麼
    # 參考https://zhuanlan.zhihu.com/p/32887066
    op = tf.add(output, 0, name='op_to_store')

    loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)  # compute cost
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)

    accuracy = tf.metrics.accuracy(  # return (acc, update_op), and create 2 local variables
        labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1), )[1]

    # 初始化網絡
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())  # the local var is for accuracy_op
    sess.run(init_op)  # initialize var in graph
    #
    # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["op_to_store"])

    for step in range(8000):  # training
        # b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
        b_x, b_y = next_batch(BATCH_SIZE)
        # b_x = b_x.reshape([BATCH_SIZE, TIME_STEP, INPUT_SIZE])
        _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
        if step % 50 == 0:  # testing23
            accuracy_ = sess.run(accuracy, {tf_x: X_test, tf_y: y_test})
            print('train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
        if step == 7999:
            saver = tf.train.Saver()
            saver.save(sess, "/home/gaofei/PycharmProjects/tf2rnnlstm/savemodel")
    print("訓練完成")


# 重新加載網絡參與運算
# sess = tf.Session()
# with gfile.FastGFile(pb_file_path+'rnnlstmmodel.pb', 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     sess.graph.as_default()
#     tf.import_graph_def(graph_def, name='') # 导入计算图
# #
# # # 需要有一个初始化的过程
# sess.run(tf.global_variables_initializer())
# # 输入初始話
# input_x = sess.graph.get_tensor_by_name('x:0')
# # input_y = sess.graph.get_tensor_by_name('y:0')
# # 輸出初始話
# output = sess.graph.get_tensor_by_name('op_to_store:0')
#
# # #加載實時數據
# ser = serial.Serial("/dev/ttyUSB0",256000,timeout = 0.01)
# ser.flushInput()
# while True:
#     currentdata = ser.readline()
#     if currentdata !=b'' :
#         # starttime= time.time()
#         currentdata = str(currentdata, 'UTF-8')
#         currentdatalist = currentdata.split('\r\n')[0]
#         currentdatalist = currentdatalist.split(",")
#         dataarray = np.array(currentdatalist,dtype='float32').reshape((-1,9))
#         newx = dataarray[:, :9]
#         test_output = sess.run(output, {input_x: newx})
#         pred_y = np.argmax(test_output, 1)
#         # endtime = time.time()
#         # print("單次運算的時間:",endtime-starttime)
#         if pred_y==0:
#             print("當前在額頭區域")
#             cv2.imshow('image', lefthead)
#             cv2.waitKey(30)
#         elif pred_y==1:
#             # print("current is" ,areaid)
#             print("當前在下頜線區域")
#             cv2.imshow('image', leftjaw)
#             cv2.waitKey(30)
#         elif pred_y==2:
#             # print("current is" ,areaid)
#             print("當前在面部區域")
#             cv2.imshow('image', leftface)
#             cv2.waitKey(30)
#         elif pred_y==3:
#             # print("current is" ,areaid)
#             print("當前在眼周區域")
#             cv2.imshow('image', lefteye)
#             cv2.waitKey(30)
#     time.sleep(0.001)
#

# print(ret)
















#
