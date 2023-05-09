import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

tf.compat.v1.set_random_seed(777)
tf.compat.v1.disable_eager_execution()

# 加載數據
def loaddata(filepath):
    all_data = np.loadtxt(filepath)
    return all_data

# 超參數設計
BATCH_SIZE =128
TIME_STEP = 1
INPUT_SIZE = 18
LR = 0.001
CLASS_NUM = 4
BasicLSTMCell_NUM = 128

# 加載所有數據
path = "/home/gaofei/PycharmProjects/tf2rnnlstm/9Data4Area/autoCatchdataandlearn/personaldata/Soarrightbi/megedData.txt"
dataSet = loaddata(path)
x = dataSet[:,:18]
y = dataSet[:,18:22]

# 區分訓練集合和驗證集合
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.25)

# 輪循抓取數據的參數
epochs_completed = 0
index_in_epoch = 0
num_examples = X_train.shape[0]

# 抓取Batch數據
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

tf_x = tf.compat.v1.placeholder(tf.float32, [None, TIME_STEP * INPUT_SIZE],name="Input")  #(128,9)     # shape(batch, 784)
image = tf.compat.v1.reshape(tf_x, [-1, TIME_STEP, INPUT_SIZE])  #(128,1,9)                 # (batch, height, width, channel)
tf_y = tf.compat.v1.placeholder(tf.int32, [None, CLASS_NUM])   #(128,4)                          # input y

# RNN
# https://blog.csdn.net/qq_44368508/article/details/126994477
# rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=BasicLSTMCell_NUM)
rnn_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=BasicLSTMCell_NUM)
#(128,28,128)
outputs, (h_c, h_n) = tf.compat.v1.nn.dynamic_rnn(
    rnn_cell,                   # cell you have chosen
    image,                      # input
    initial_state=None,         # the initial hidden state
    dtype=tf.float32,           # must given if set initial_state = None
    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
)
output = tf.compat.v1.layers.dense(outputs[:, -1, :], CLASS_NUM)              # output based on the last output step
Output = tf.add(output, 0, name='Output')

loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.compat.v1.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.compat.v1.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.compat.v1.Session()
init_op = tf.compat.v1.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph

for step in range(80000):    # training
    # b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    b_x, b_y = next_batch(BATCH_SIZE)
    # b_x = b_x.reshape([BATCH_SIZE, TIME_STEP, INPUT_SIZE])
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    if step % 50 == 0:      # testing23
        accuracy_ = sess.run(accuracy, {tf_x: X_test, tf_y: y_test})
        print('train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
    # 添加保存數據模塊
    if step == 79999:
        saver = tf.compat.v1.train.Saver()
        # saver.save(sess, save_path="/home/gaofei/PycharmProjects/tf2rnnlstm/rnnlstmtf2for7area/Matrix")
        saver.save(sess, save_path="/home/gaofei/PycharmProjects/tf2rnnlstm/9Data4Area/autoCatchdataandlearn/personnalmodel/rightface/Matrixrightbi")


