import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

class TrainModel:
    BATCH_SIZE = 64
    # BATCH_SIZE = 128
    TIME_STEP = 1
    INPUT_SIZE = 18
    LR = 0.0001
    CLASS_NUM = 4
    BasicLSTMCell_NUM = 128

    epochs_completed = 0
    index_in_epoch = 0
    # num_examples = X_train.shape[0]
    num_examples = 0

    X_train = None
    X_test = None
    y_train = None
    y_test = None

    def __init__(self,datafilePath,modelsavepath):
        # self.filePath =".\\Data\\leftdata\\"
        # self.savepath = ".\\Model\\leftmodel\\Matrixleftbi"
        self.filePath = datafilePath
        self.savepath = modelsavepath

    def merge4(self):
        area0 = np.loadtxt(self.filePath + "area0.txt")
        area1 = np.loadtxt(self.filePath + "area1.txt")
        area2 = np.loadtxt(self.filePath + "area2.txt")
        area3 = np.loadtxt(self.filePath + "area3.txt")

        mergeddata = np.r_[area0, area1]
        mergeddata = np.r_[mergeddata, area2]
        mergeddata = np.r_[mergeddata, area3]
        print(mergeddata.shape)

        np.savetxt(self.filePath + "megedData.txt", mergeddata)

    def loaddata(self):
        all_data = np.loadtxt(self.filePath+ "megedData.txt")
        return all_data

    def next_batch(self):
        start = self.index_in_epoch
        self.index_in_epoch += self.BATCH_SIZE

        # when all trainig data have been already used, it is reorder randomly
        if self.index_in_epoch > self.num_examples:
            # finished epoch
            self.epochs_completed += 1
            # shuffle the data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            X_train = self.X_train[perm]
            y_train = self.y_train[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = self.BATCH_SIZE
            assert self.BATCH_SIZE <= self.num_examples
        end = self.index_in_epoch
        return self.X_train[start:end], self.y_train[start:end]

    def trainmodel(self):
        print("train model")
        self.merge4()
        dataSet = self.loaddata()
        x = dataSet[:, :18]
        y = dataSet[:, 18:22]
        # x = dataSet[:, :9]
        # y = dataSet[:, 9:13]
        # 區分訓練集合和驗證集合
        self.X_train, self.X_test, self.y_train, \
            self.y_test = train_test_split(x, y, test_size=0.25)
        self.num_examples = self.X_train.shape[0]

        # 构建网络
        tf.compat.v1.set_random_seed(777)
        tf.compat.v1.disable_eager_execution()

        # 重置模型图
        tf.compat.v1.reset_default_graph()

        tf_x = tf.compat.v1.placeholder(tf.float32, [None, self.TIME_STEP * self.INPUT_SIZE], name="Input")  # (none,9)
        image = tf.compat.v1.reshape(tf_x, [-1, self.TIME_STEP, self.INPUT_SIZE])  # (128,1,9)
        tf_y = tf.compat.v1.placeholder(tf.int32, [None, self.CLASS_NUM])  # (128,4)

        # RNN
        # https://blog.csdn.net/qq_44368508/article/details/126994477
        rnn_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=self.BasicLSTMCell_NUM)
        # (128,28,128)
        outputs, (h_c, h_n) = tf.compat.v1.nn.dynamic_rnn(
            rnn_cell,
            image,
            initial_state=None,
            dtype=tf.float32,
            time_major=False,
        )
        output = tf.compat.v1.layers.dense(outputs[:, -1, :], self.CLASS_NUM)
        Output = tf.add(output, 0, name='Output')

        loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
        train_op = tf.compat.v1.train.AdamOptimizer(self.LR).minimize(loss)

        accuracy = tf.compat.v1.metrics.accuracy(
            labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1), )[1]

        sess = tf.compat.v1.Session()
        init_op = tf.compat.v1.group(tf.compat.v1.global_variables_initializer(),
                                     tf.compat.v1.local_variables_initializer())
        sess.run(init_op)

        for step in range(8000):
            b_x, b_y = self.next_batch()
            _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
            if step % 50 == 0:
                accuracy_ = sess.run(accuracy, {tf_x: self.X_test, tf_y: self.y_test})
                print('train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
            # 添加保存數據模塊
            if step == 7999:
                saver = tf.compat.v1.train.Saver()
                saver.save(sess, save_path=self.savepath)


if __name__ =="__main__":
    filePath ="./Data/leftdata/"
    savepath = "./Model/leftmodel/Matrixleftbi"
    traninleftdemo = TrainModel(filePath,savepath)
    traninleftdemo.trainmodel()

    filePath ="./Data/rightdata/"
    savepath = "./Model/rightmodel/Matrixrightbi"
    traninrightdemo = TrainModel(filePath,savepath)
    traninrightdemo.trainmodel()


