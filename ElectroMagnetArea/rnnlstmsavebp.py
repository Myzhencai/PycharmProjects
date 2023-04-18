import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import serial # 导入串口包
import time # 导入时间包
import cv2
from tensorflow import keras
from tensorflow.python.framework import graph_util
import tensorflow as tf
from tensorflow.saved_model.signature_def_utils import predict_signature_def
from tensorflow.saved_model import tag_constants


# 加載數據
def loaddata(filepath):
    all_data = np.loadtxt(filepath)
    return all_data


tf.set_random_seed(1)
np.random.seed(1)

# Hyper Parameters
BATCH_SIZE =128
TIME_STEP = 1          # rnn time step / image height
INPUT_SIZE = 9         # rnn input size / image width
LR = 0.01               # learning rate
CLASS_NUM = 4
BasicLSTMCell_NUM = 128

# 加載數據
path = "/home/gaofei/PycharmProjects/ElectroMagnetArea/Data/HalfFace7/megedData.txt"
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


tf_x = tf.placeholder(tf.float32, [None, TIME_STEP * INPUT_SIZE],name='Inputs')  #(128,9)     # shape(batch, 784)
image = tf.reshape(tf_x, [-1, TIME_STEP, INPUT_SIZE])  #(128,1,9)                 # (batch, height, width, channel)
tf_y = tf.placeholder(tf.int32, [None, CLASS_NUM],name = "Prediction")   #(128,4)                          # input y

# RNN
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=BasicLSTMCell_NUM)
#(128,28,128)
outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
    rnn_cell,                   # cell you have chosen
    image,                      # input
    initial_state=None,         # the initial hidden state
    dtype=tf.float32,           # must given if set initial_state = None
    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
)
output = tf.layers.dense(outputs[:, -1, :], CLASS_NUM)              # output based on the last output step

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph

for step in range(800):    # training
    # b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    b_x, b_y = next_batch(BATCH_SIZE)
    # b_x = b_x.reshape([BATCH_SIZE, TIME_STEP, INPUT_SIZE])
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    if step % 50 == 0:      # testing23
        accuracy_ = sess.run(accuracy, {tf_x: X_test, tf_y: y_test})
        print('train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
    if(step == 799):
        builder = tf.saved_model.builder.SavedModelBuilder("./saved_model")
        signature = predict_signature_def(inputs={'Inputs': tf_x},
                                          outputs={'Prediction': tf_y})
        builder.add_meta_graph_and_variables(sess=sess,
                                             tags=[tag_constants.SERVING])
        builder.save()
        # https://blog.csdn.net/qq_27825451/article/details/105866464
        # tf.saved_model.simple_save(sess, "./saved_model", inputs={"Inputs": tf_x}, outputs={"Prediction": tf_y})
        # print("model has saved,model format is saved_model !")

# 保存模型從輸入到輸出https://blog.csdn.net/qq_37791134/article/details/104758416
# graph_def = tf.get_default_graph().as_graph_def()
# output_graph_def = graph_util.convert_variables_to_constants(
#     sess,
#     graph_def,
#     ['Prediction']
# )


