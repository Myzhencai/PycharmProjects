
import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_eager_execution()


sess=tf.Session()
#先加载图和参数变量
saver = tf.train.import_meta_graph('/home/gaofei/PycharmProjects/ElectroMagnetArea/demodata/ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('/home/gaofei/PycharmProjects/ElectroMagnetArea/demodata'))

# 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
graph = tf.get_default_graph()
input_x = sess.graph.get_tensor_by_name('x:0')
output = sess.graph.get_tensor_by_name('op_to_store:0')
# #加載實時數據
while True:
    newx = np.array([124,546,234,243,564,134,535,643,674]).reshape((-1,9))
    test_output = sess.run(output, {input_x: newx})
    pred_y = np.argmax(test_output, 1)
    print("pred_y: ",pred_y)
