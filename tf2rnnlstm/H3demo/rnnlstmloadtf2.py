import tensorflow.compat.v1 as tf
import numpy as np
import time

tf.disable_eager_execution()


sess=tf.Session()
#先加载图和参数变量
saver = tf.train.import_meta_graph('/home/gaofei/PycharmProjects/tf2rnnlstm/savemodel/rnnlstmtf2.meta')
saver.restore(sess, tf.train.latest_checkpoint('/home/gaofei/PycharmProjects/tf2rnnlstm/savemodel'))

# 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
graph = tf.get_default_graph()
input_x = sess.graph.get_tensor_by_name('Input:0')
output = sess.graph.get_tensor_by_name('Output:0')
# #加載實時數據
while True:
    starttime = time.time()
    newx = np.array([124,546,234,243,564,134,535,643,674]).reshape((-1,9))
    test_output = sess.run(output, {input_x: newx})
    pred_y = np.argmax(test_output, 1)
    endtime = time.time()
    print("used time :",endtime-starttime)
    print("pred_y: ",pred_y)