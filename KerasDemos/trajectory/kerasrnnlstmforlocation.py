# 借鑑https://github.com/eastmountyxz/AI-for-Keras/blob/master/blog14-LSTM-RNN/blog14-lstm.py
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ----------------------------------定义参数----------------------------------
BATCH_INDEX = 0
TIME_STEPS = 1
BATCH_SIZE = 50  # BATCH数量
INPUT_SIZE = 9  # 输入一个值
OUTPUT_SIZE = 3 # 输出一个值
CELL_SIZE = 100  # Cell数量
LR = 0.006
BATCH_START_TEST = 0

path = "/home/gaofei/PycharmProjects/ElectroMagnetArea/SoarFacedata/megedData.txt"
dataSet = np.loadtxt(path)
x = dataSet[:, :9]
y = dataSet[:, 9:12]

# 區分訓練集合和驗證集合
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# 获取批量数据
def get_batch():
    global BATCH_INDEX, TIME_STEPS
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :, :]
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :, :]

    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX
    return X_batch,Y_batch


# ----------------------------------LSTM RNN----------------------------------
class LSTMRNN(object):
    # 初始化操作
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size

        # TensorBoard可视化操作使用name_scope
        with tf.name_scope('inputs'):  # 输出变量
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.variable_scope('in_hidden'):  # 输入层
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):  # 处理层
            self.add_cell()
        with tf.variable_scope('out_hidden'):  # 输出层
            self.add_output_layer()
        with tf.name_scope('cost'):  # 误差
            self.compute_cost()
        with tf.name_scope('train'):  # 训练
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    # --------------------------------定义核心三层结构-----------------------------
    # 输入层
    def add_input_layer(self, ):
        # 定义输入层xs变量 将xs三维数据转换成二维
        # [None, n_steps, input_size] => (batch*n_step, in_size)
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')
        # 定义输入权重 (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # 定义输入偏置 (cell_size, )
        bs_in = self._bias_variable([self.cell_size, ])
        # 定义输出y变量 二维形状 (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # 返回结果形状转变为三维
        # l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    # cell层
    def add_cell(self):
        # 选择BasicLSTMCell模型
        # forget初始偏置为1.0(初始时不希望forget) 随着训练深入LSTM会选择性忘记
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        # 设置initial_state全为0 可视化操作用name_scope
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        # RNN循环 每一步的输出都存储在cell_outputs序列中 cell_final_state为最终State并传入下一个batch中
        # 常规RNN只有m_state LSTM包括c_state和m_state
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

    # 输出层 (类似输入层)
    def add_output_layer(self):
        # 转换成二维 方能使用W*X+B
        # shape => (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # 返回预测结果
        # shape => (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

    # --------------------------------定义误差计算函数-----------------------------
    def compute_cost(self):
        # 使用seq2seq序列到序列模型
        # tf.nn.seq2seq.sequence_loss_by_example()
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.msr_error,
            name='losses'
        )
        # 最终得到batch的总cost 它是一个数字
        with tf.name_scope('average_cost'):
            # 整个TensorFlow的loss求和 再除以batch size
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    # 该函数用于计算
    # 相当于msr_error(self, y_pre, y_target) return tf.square(tf.sub(y_pre, y_target))
    def msr_error(self, logits, labels):
        # tempresult = tf.subtract(logits, labels)
        # tempresult = tf.square(tf.subtract(logits, labels))
        # tempresult2 = tf.reshape(tempresult, [-1,self.output_size,self.batch_size])
        # return tf.square(tf.subtract(logits, labels))
        return tf.reshape(tf.square(tf.subtract(logits, labels)), [-1,self.output_size,self.batch_size])

    # 误差计算
    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1., )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    # 偏置计算
    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


# ----------------------------------主函数 训练和预测----------------------------------
if __name__ == '__main__':
    # 定义模型并初始化
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    sess.run(tf.initialize_all_variables())
    # Tensorboard可视化展现神经网络结果

    # ------------------------------RNN学习-------------------------------------
    # 交互模式启动
    plt.ion()
    plt.show()

    # 训练模型
    for i in range(200):
        # 用seq预测res (序列-seq 结果-res 输入-xs)
        Input, Output = get_batch()
        # 第一步赋值 之后会更新cell_init_state
        if i == 0:
            feed_dict = {
                model.xs: Input,
                model.ys: Output,
                # create initial state (前面cell_init_state已初始化state)
            }
        else:
            feed_dict = {
                model.xs: Input,
                model.ys: Output,
                model.cell_init_state: state
                # use last state as the initial state for this run
            }

        # state为final_state
        _, cost, state, pred = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred],
            feed_dict=feed_dict)

        # plotting
        # 获取第一批数据xs[0,:] 获取0到20区间的预测数据pred.flatten()[:TIME_STEPS]
        # plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')
        # plt.ylim((-1.2, 1.2))
        # plt.draw()
        # plt.pause(0.3)

        # 每隔20步输出结果
        if i % 20 == 0:
            print('cost: ', round(cost, 4))
            # result = sess.run(merged, feed_dict)
            # writer.add_summary(result, i)
#


# # -*- coding: utf-8 -*-
# """
# Created on Thu Jan  9 20:44:56 2020
# @author: xiuzhang Eastmount CSDN
# """
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
#
# # ----------------------------------定义参数----------------------------------
# BATCH_START = 0
# TIME_STEPS = 20
# BATCH_SIZE = 50  # BATCH数量
# INPUT_SIZE = 1  # 输入一个值
# OUTPUT_SIZE = 1  # 输出一个值
# CELL_SIZE = 10  # Cell数量
# LR = 0.006
# BATCH_START_TEST = 0
#
#
# # 获取批量数据
# def get_batch():
#     global BATCH_START, TIME_STEPS
#     # xs shape (50batch, 20steps)
#     xs = np.arange(BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10 * np.pi)
#     seq = np.sin(xs)
#     res = np.cos(xs)
#     BATCH_START += TIME_STEPS
#
#     # 显示原始曲线
#     # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
#     # plt.show()
#
#     # 返回序列seq 结果res 输入xs
#     return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]
#
#
# # ----------------------------------LSTM RNN----------------------------------
# class LSTMRNN(object):
#     # 初始化操作
#     def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
#         self.n_steps = n_steps
#         self.input_size = input_size
#         self.output_size = output_size
#         self.cell_size = cell_size
#         self.batch_size = batch_size
#
#         # TensorBoard可视化操作使用name_scope
#         with tf.name_scope('inputs'):  # 输出变量
#             self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
#             self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
#         with tf.variable_scope('in_hidden'):  # 输入层
#             self.add_input_layer()
#         with tf.variable_scope('LSTM_cell'):  # 处理层
#             self.add_cell()
#         with tf.variable_scope('out_hidden'):  # 输出层
#             self.add_output_layer()
#         with tf.name_scope('cost'):  # 误差
#             self.compute_cost()
#         with tf.name_scope('train'):  # 训练
#             self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)
#
#     # --------------------------------定义核心三层结构-----------------------------
#     # 输入层
#     def add_input_layer(self, ):
#         # 定义输入层xs变量 将xs三维数据转换成二维
#         # [None, n_steps, input_size] => (batch*n_step, in_size)
#         l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')
#         # 定义输入权重 (in_size, cell_size)
#         Ws_in = self._weight_variable([self.input_size, self.cell_size])
#         # 定义输入偏置 (cell_size, )
#         bs_in = self._bias_variable([self.cell_size, ])
#         # 定义输出y变量 二维形状 (batch * n_steps, cell_size)
#         with tf.name_scope('Wx_plus_b'):
#             l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
#         # 返回结果形状转变为三维
#         # l_in_y ==> (batch, n_steps, cell_size)
#         self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')
#
#     # cell层
#     def add_cell(self):
#         # 选择BasicLSTMCell模型
#         # forget初始偏置为1.0(初始时不希望forget) 随着训练深入LSTM会选择性忘记
#         lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
#         # 设置initial_state全为0 可视化操作用name_scope
#         with tf.name_scope('initial_state'):
#             self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
#         # RNN循环 每一步的输出都存储在cell_outputs序列中 cell_final_state为最终State并传入下一个batch中
#         # 常规RNN只有m_state LSTM包括c_state和m_state
#         self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
#             lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)
#
#     # 输出层 (类似输入层)
#     def add_output_layer(self):
#         # 转换成二维 方能使用W*X+B
#         # shape => (batch * steps, cell_size)
#         l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
#         Ws_out = self._weight_variable([self.cell_size, self.output_size])
#         bs_out = self._bias_variable([self.output_size, ])
#         # 返回预测结果
#         # shape => (batch * steps, output_size)
#         with tf.name_scope('Wx_plus_b'):
#             self.pred = tf.matmul(l_out_x, Ws_out) + bs_out
#
#     # --------------------------------定义误差计算函数-----------------------------
#     def compute_cost(self):
#         # 使用seq2seq序列到序列模型
#         # tf.nn.seq2seq.sequence_loss_by_example()
#         losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
#             [tf.reshape(self.pred, [-1], name='reshape_pred')],
#             [tf.reshape(self.ys, [-1], name='reshape_target')],
#             [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
#             average_across_timesteps=True,
#             softmax_loss_function=self.msr_error,
#             name='losses'
#         )
#         # 最终得到batch的总cost 它是一个数字
#         with tf.name_scope('average_cost'):
#             # 整个TensorFlow的loss求和 再除以batch size
#             self.cost = tf.div(
#                 tf.reduce_sum(losses, name='losses_sum'),
#                 self.batch_size,
#                 name='average_cost')
#             tf.summary.scalar('cost', self.cost)
#
#     # 该函数用于计算
#     # 相当于msr_error(self, y_pre, y_target) return tf.square(tf.sub(y_pre, y_target))
#     def msr_error(self, logits, labels):
#         return tf.square(tf.subtract(logits, labels))
#
#     # 误差计算
#     def _weight_variable(self, shape, name='weights'):
#         initializer = tf.random_normal_initializer(mean=0., stddev=1., )
#         return tf.get_variable(shape=shape, initializer=initializer, name=name)
#
#     # 偏置计算
#     def _bias_variable(self, shape, name='biases'):
#         initializer = tf.constant_initializer(0.1)
#         return tf.get_variable(name=name, shape=shape, initializer=initializer)
#
#
# # ----------------------------------主函数 训练和预测----------------------------------
# if __name__ == '__main__':
#     # 定义模型并初始化
#     model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
#     sess = tf.Session()
#     merged = tf.summary.merge_all()
#     writer = tf.summary.FileWriter("logs", sess.graph)
#     sess.run(tf.initialize_all_variables())
#     # Tensorboard可视化展现神经网络结果
#
#     # ------------------------------RNN学习-------------------------------------
#     # 交互模式启动
#     plt.ion()
#     plt.show()
#
#     # 训练模型
#     for i in range(200):
#         # 用seq预测res (序列-seq 结果-res 输入-xs)
#         seq, res, xs = get_batch()
#         # 第一步赋值 之后会更新cell_init_state
#         if i == 0:
#             feed_dict = {
#                 model.xs: seq,
#                 model.ys: res,
#                 # create initial state (前面cell_init_state已初始化state)
#             }
#         else:
#             feed_dict = {
#                 model.xs: seq,
#                 model.ys: res,
#                 model.cell_init_state: state
#                 # use last state as the initial state for this run
#             }
#
#         # state为final_state
#         _, cost, state, pred = sess.run(
#             [model.train_op, model.cost, model.cell_final_state, model.pred],
#             feed_dict=feed_dict)
#
#         # plotting
#         # 获取第一批数据xs[0,:] 获取0到20区间的预测数据pred.flatten()[:TIME_STEPS]
#         plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')
#         plt.ylim((-1.2, 1.2))
#         plt.draw()
#         plt.pause(0.3)
#
#         # 每隔20步输出结果
#         if i % 20 == 0:
#             print('cost: ', round(cost, 4))
#             # result = sess.run(merged, feed_dict)
#             # writer.add_summary(result, i)