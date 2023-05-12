import os
import socket
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# 左右臉部數據抓取部分
datasaverleft = []
datasaverright = []
def getleftdata(areaid,savetest,Savepath,dataenoughnum):
    datasaverleft.clear()
    currentnum = 0
    while currentnum < dataenoughnum:
        # socket 讀取一行
        # 返回信息已經接到了數據
        client.send('leftdata'.encode())
        currentdataleft = client.makefile().readline()

        # 處理數據
        currentdatasaverleftleftold = currentdataleft.split('\n')[0]
        currentdatasaverleftleft = currentdatasaverleftleftold.split(",")
        inversecurrentdatasaverleftleft =currentdatasaverleftleftold.split(",")
        inversecurrentdatasaverleftleft.reverse()
        currentdatasaverleftleft = currentdatasaverleftleft + inversecurrentdatasaverleftleft
        dataarrayleft = np.array(currentdatasaverleftleft, dtype='float32').reshape((-1, 18))
        if areaid ==0:
            print("左邊額頭數據", currentnum)
            areaarray = np.array([[1,0,0,0]])
            currentdataAndarea = np.c_[dataarrayleft,areaarray]
            datasaverleft.append(currentdataAndarea[0])
            dataarray = np.array(datasaverleft,dtype='float32').reshape((-1,22))
            # print(dataarray)
            if savetest:
                np.savetxt(Savepath+"area{0}test.txt".format(areaid),dataarray)
            else:
                np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)
            currentnum +=1
        elif areaid ==1:
            print("對應左下頜", currentnum)
            areaarray = np.array([[0, 1, 0, 0]])
            currentdataAndarea = np.c_[dataarrayleft, areaarray]
            datasaverleft.append(currentdataAndarea[0])
            dataarray = np.array(datasaverleft, dtype='float32').reshape((-1, 22))
            if savetest:
                np.savetxt(Savepath + "area{0}test.txt".format(areaid), dataarray)
            else:
                np.savetxt(Savepath + "area{0}.txt".format(areaid), dataarray)
            currentnum += 1
        elif areaid ==2:
            print("對應左邊面部", currentnum)
            areaarray = np.array([[0, 0, 1, 0]])
            currentdataAndarea = np.c_[dataarrayleft, areaarray]
            datasaverleft.append(currentdataAndarea[0])
            dataarray = np.array(datasaverleft, dtype='float32').reshape((-1, 22))
            if savetest:
                np.savetxt(Savepath + "area{0}test.txt".format(areaid), dataarray)
            else:
                np.savetxt(Savepath + "area{0}.txt".format(areaid), dataarray)
            currentnum += 1
        elif areaid ==3:
            print("左邊眼周", currentnum)
            areaarray = np.array([[0, 0, 0, 1]])
            currentdataAndarea = np.c_[dataarrayleft, areaarray]
            datasaverleft.append(currentdataAndarea[0])
            dataarray = np.array(datasaverleft, dtype='float32').reshape((-1, 22))
            if savetest:
                np.savetxt(Savepath + "area{0}test.txt".format(areaid), dataarray)
            else:
                np.savetxt(Savepath + "area{0}.txt".format(areaid), dataarray)
            currentnum += 1

    return areaid


def getrightdata(areaid,savetest,Savepath,dataenoughnum):
    datasaverleft.clear()
    currentnum = 0
    while currentnum < dataenoughnum:
        # socket 讀取一行
        client.send('rightdata'.encode())
        currentdataleft = client.makefile().readline()
        # 返回信息已經接到了數據

        # 處理數據
        currentdatasaverleftleftold = currentdataleft.split('\n')[0]
        currentdatasaverleftleft = currentdatasaverleftleftold.split(",")
        inversecurrentdatasaverleftleft =currentdatasaverleftleftold.split(",")
        inversecurrentdatasaverleftleft.reverse()
        currentdatasaverleftleft = currentdatasaverleftleft + inversecurrentdatasaverleftleft
        dataarrayleft = np.array(currentdatasaverleftleft, dtype='float32').reshape((-1, 18))
        if areaid ==0:
            print("右邊額頭數據", currentnum)
            areaarray = np.array([[1,0,0,0]])
            currentdataAndarea = np.c_[dataarrayleft,areaarray]
            datasaverleft.append(currentdataAndarea[0])
            dataarray = np.array(datasaverleft,dtype='float32').reshape((-1,22))
            # print(dataarray)
            if savetest:
                np.savetxt(Savepath+"area{0}test.txt".format(areaid),dataarray)
            else:
                np.savetxt(Savepath+"area{0}.txt".format(areaid), dataarray)
            currentnum +=1
        elif areaid ==1:
            print("對應右下頜", currentnum)
            areaarray = np.array([[0, 1, 0, 0]])
            currentdataAndarea = np.c_[dataarrayleft, areaarray]
            datasaverleft.append(currentdataAndarea[0])
            dataarray = np.array(datasaverleft, dtype='float32').reshape((-1, 22))
            if savetest:
                np.savetxt(Savepath + "area{0}test.txt".format(areaid), dataarray)
            else:
                np.savetxt(Savepath + "area{0}.txt".format(areaid), dataarray)
            currentnum += 1
        elif areaid ==2:
            print("對應右邊面部", currentnum)
            areaarray = np.array([[0, 0, 1, 0]])
            currentdataAndarea = np.c_[dataarrayleft, areaarray]
            datasaverleft.append(currentdataAndarea[0])
            dataarray = np.array(datasaverleft, dtype='float32').reshape((-1, 22))
            if savetest:
                np.savetxt(Savepath + "area{0}test.txt".format(areaid), dataarray)
            else:
                np.savetxt(Savepath + "area{0}.txt".format(areaid), dataarray)
            currentnum += 1
        elif areaid ==3:
            print("右邊眼周", currentnum)
            areaarray = np.array([[0, 0, 0, 1]])
            currentdataAndarea = np.c_[dataarrayleft, areaarray]
            datasaverleft.append(currentdataAndarea[0])
            dataarray = np.array(datasaverleft, dtype='float32').reshape((-1, 22))
            if savetest:
                np.savetxt(Savepath + "area{0}test.txt".format(areaid), dataarray)
            else:
                np.savetxt(Savepath + "area{0}.txt".format(areaid), dataarray)
            currentnum += 1

    return areaid

def autotrainmodel(datapath):
    tf.compat.v1.set_random_seed(777)
    tf.compat.v1.disable_eager_execution()
    # 加載數據
    def loaddata(filepath):
        all_data = np.loadtxt(filepath)
        return all_data

    # 超參數
    BATCH_SIZE =128
    TIME_STEP = 1
    INPUT_SIZE = 18
    LR = 0.001
    CLASS_NUM = 4
    BasicLSTMCell_NUM = 128

    # 加載所有數據
    path = datapath
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

    tf_x = tf.compat.v1.placeholder(tf.float32, [None, TIME_STEP * INPUT_SIZE],name="Input")  #(none,9)
    image = tf.compat.v1.reshape(tf_x, [-1, TIME_STEP, INPUT_SIZE])  #(128,1,9)
    tf_y = tf.compat.v1.placeholder(tf.int32, [None, CLASS_NUM])   #(128,4)

    # RNN
    # https://blog.csdn.net/qq_44368508/article/details/126994477
    rnn_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=BasicLSTMCell_NUM)
    #(128,28,128)
    outputs, (h_c, h_n) = tf.compat.v1.nn.dynamic_rnn(
        rnn_cell,
        image,
        initial_state=None,
        dtype=tf.float32,
        time_major=False,
    )
    output = tf.compat.v1.layers.dense(outputs[:, -1, :], CLASS_NUM)
    Output = tf.add(output, 0, name='Output')

    loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
    train_op = tf.compat.v1.train.AdamOptimizer(LR).minimize(loss)

    accuracy = tf.compat.v1.metrics.accuracy(
        labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

    sess = tf.compat.v1.Session()
    init_op = tf.compat.v1.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
    sess.run(init_op)

    for step in range(8000):
        b_x, b_y = next_batch(BATCH_SIZE)
        _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
        if step % 50 == 0:
            accuracy_ = sess.run(accuracy, {tf_x: X_test, tf_y: y_test})
            print('train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
        # 添加保存數據模塊
        if step == 7999:
            saver = tf.compat.v1.train.Saver()
            saver.save(sess, save_path="/home/rer/model/leftmodel/Matrixleftbi")


# 融合數據
def merge4(filePath):
    area0 = np.loadtxt(filePath+"area0.txt")
    area1 = np.loadtxt(filePath+"area1.txt")
    area2 = np.loadtxt(filePath+"area2.txt")
    area3 = np.loadtxt(filePath+"area3.txt")

    mergeddata = np.r_[area0, area1]
    mergeddata = np.r_[mergeddata, area2]
    mergeddata = np.r_[mergeddata, area3]
    print(mergeddata.shape)

    np.savetxt(filePath+"megedData.txt", mergeddata)


#
# if __name__ =="__main__":
#     # filePath ="/home/gaofei/PycharmProjects/ElectroMagnetArea/SoarFacedata/"
#     # filePath ="/home/gaofei/PycharmProjects/ElectroMagnetArea/SoarFacedata7new/"
#     # filePath ="/home/gaofei/PycharmProjects/tf2rnnlstm/DatafromH3/7Areadata/"
#     # filePath ="/home/gaofei/PycharmProjects/tf2rnnlstm/H3getdata/data/left/"
#     # filePath ="/home/gaofei/PycharmProjects/tf2rnnlstm/9Data4Area/data/rightbi/"




# 鏈接H3並構建聯系部分
flag = True
# 生成socket对象
client = socket.socket()
# ipaddress = input("請輸入電腦網絡的ip地址（更具命令行得到的結果）：")
ipaddress = '172.16.2.52'
# ipaddress = '192.168.43.23'
# 链接要链接的ip和port（端口）
client.connect((ipaddress, 6868))
print("鏈接上了Matrix")


# flag = True
# # 生成socket对象
# server = socket.socket()
# # 绑定ip和端口
# server.bind(('172.16.2.52', 6868))
# # 监听绑定的端口
# server.listen()
# # 方便识别打印一个我在等待
# print("等待Matrix鏈接...")
# conn, addr = server.accept()
# print('成功鏈接Matrix')


finishleft = False
finishright = False
enoughNum = 600
arealist =[0,1,2,3]

# 清除路徑下的原始數據
savePathlist = ["./data/leftdata/","./data/rightdata/"]

for savepath in savePathlist:
    dir = savepath
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

# 进入循环
while True:
    # 打開左邊
    if finishleft is False:
        openleft = input("請問打開左邊的串口進行數據收集麼？（yes or no）:").strip()
        if openleft =='yes':
            # 發送信號打開左邊串口並監聽H3是否打開左邊的串口
            client.send('openleft'.encode())
            while True:
                openedleft = client.recv(1024).decode()
                if openedleft == 'openedleft':
                    print("左邊的串口已經打開並在傳輸數據 ")
                    for areaid in arealist:
                        if areaid == 0:
                            while input("請將美容儀器開機放置在額頭區域（左），完成後請按Enter鍵 :") != '':
                                print("重新輸入")
                            print("左邊額頭數據")
                            finishedid = getleftdata(areaid,False, "./data/leftdata/", enoughNum)
                            # print(finishedid)
                        if areaid == 1 and finishedid == 0:
                            while input("請將美容儀器開機放置在下頜線區域（左），完成後請按Enter鍵 :") != '':
                                print("重新輸入")
                            print("對應左下頜")
                            finishedid = getleftdata(areaid,False, "./data/leftdata/", enoughNum)
                            # print(finishedid)
                        if areaid == 2 and finishedid == 1:
                            while input("請將美容儀器開機放置在臉部區域（左），完成後請按Enter鍵 :") != '':
                                print("重新輸入")
                            print("對應左邊面部")
                            finishedid = getleftdata(areaid,False, "./data/leftdata/", enoughNum)
                        if areaid == 3 and finishedid == 2:
                            while input("請將美容儀器開機放置在眼周區域（左），完成後請按Enter鍵 :") != '':
                                print("重新輸入")
                            print("左邊眼周")
                            finishedid = getleftdata(areaid,False, "./data/leftdata/", enoughNum)
                        if finishedid == 3:
                            print("完成左邊臉部的數據採集")
                    client.send('finishleft'.encode())
                    closedleft = client.recv(1024).decode()
                    if closedleft == "closedleft":
                        finishleft = True
                        # 退出循環
                        break
        elif openleft =='no':
            print("不進行左邊sensor的採集")
        else:
            print("錯誤的輸入，請稍後重新輸入")

    if finishright is False:
        openright = input("請問打開右邊的串口進行數據收集麼？（yes or no）:").strip()
        if openright == 'yes':
            # 發送信號打開左邊串口並監聽是否打開
            client.send('openright'.encode())
            while True:
                openedlright = client.recv(1024).decode()
                if openedlright == 'openedright':
                    print("右邊的串口已經打開並在傳輸數據 ")
                    for areaid in arealist:
                        if areaid == 0:
                            while input("請將美容儀器開機放置在額頭區域（右），完成後請按Enter鍵 :") != '':
                                print("重新輸入")
                            print("右邊額頭數據")
                            finishedid = getrightdata(areaid,False, "./data/rightdata/", enoughNum)
                            # print(finishedid)
                        if areaid == 1 and finishedid == 0:
                            while input("請將美容儀器開機放置在下頜線區域（右），完成後請按Enter鍵 :") != '':
                                print("重新輸入")
                            print("對應右下頜")
                            finishedid = getrightdata(areaid,False, "./data/rightdata/", enoughNum)
                            # print(finishedid)
                        if areaid == 2 and finishedid == 1:
                            while input("請將美容儀器開機放置在臉部區域（右），完成後請按Enter鍵 :") != '':
                                print("重新輸入")
                            print("對應右邊面部")
                            finishedid = getrightdata(areaid,False, "./data/rightdata/", enoughNum)
                        if areaid == 3 and finishedid == 2:
                            while input("請將美容儀器開機放置在眼周區域（右），完成後請按Enter鍵 :") != '':
                                print("重新輸入")
                            print("右邊眼周")
                            finishedid = getrightdata(areaid,False, "./data/rightdata/", enoughNum)
                        if finishedid == 3:
                            print("完成右邊臉部的數據採集")
                    client.send('finishright'.encode())
                    closedright = client.recv(1024).decode()
                    if closedright == "closedright":
                       finishright = True
                    # 退出循環
                       break
        elif openright == 'no':
            print("不進行右邊sensor的採集")
        else:
            print("錯誤的輸入，請稍後重新輸入")

    if openright == 'no' and openleft =='no':
        print("不記錄任何數據")
        client.send('closeall'.encode())
        client.shutdown(2)
        client.close()
        break
    if openright == 'yes' and openleft =='yes':
        print("完成了左右面部數據採集關閉程序")
        client.send('closeall'.encode())
        client.shutdown(2)
        client.close()
        break

# 自動訓練模型部分
filePath ="/home/gaofei/PycharmProjects/tf2rnnlstm/socket/data/leftdata/"
merge4(filePath)
# autotrainmodel(datapath = "/home/gaofei/PycharmProjects/tf2rnnlstm/socket/data/leftdata/megedData.txt")
tf.compat.v1.set_random_seed(777)
tf.compat.v1.disable_eager_execution()
# 加載數據
def loaddata(filepath):
    all_data = np.loadtxt(filepath)
    return all_data

# 超參數
BATCH_SIZE =128
TIME_STEP = 1
INPUT_SIZE = 18
LR = 0.001
CLASS_NUM = 4
BasicLSTMCell_NUM = 128

# 加載所有數據
path = "/home/gaofei/PycharmProjects/tf2rnnlstm/socket/data/leftdata/megedData.txt"
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

tf_x = tf.compat.v1.placeholder(tf.float32, [None, TIME_STEP * INPUT_SIZE],name="Input")  #(none,9)
image = tf.compat.v1.reshape(tf_x, [-1, TIME_STEP, INPUT_SIZE])  #(128,1,9)
tf_y = tf.compat.v1.placeholder(tf.int32, [None, CLASS_NUM])   #(128,4)

# RNN
# https://blog.csdn.net/qq_44368508/article/details/126994477
rnn_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=BasicLSTMCell_NUM)
#(128,28,128)
outputs, (h_c, h_n) = tf.compat.v1.nn.dynamic_rnn(
    rnn_cell,
    image,
    initial_state=None,
    dtype=tf.float32,
    time_major=False,
)
output = tf.compat.v1.layers.dense(outputs[:, -1, :], CLASS_NUM)
Output = tf.add(output, 0, name='Output')

loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
train_op = tf.compat.v1.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.compat.v1.metrics.accuracy(
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.compat.v1.Session()
init_op = tf.compat.v1.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
sess.run(init_op)

for step in range(8000):
    b_x, b_y = next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    if step % 50 == 0:
        accuracy_ = sess.run(accuracy, {tf_x: X_test, tf_y: y_test})
        print('train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
    # 添加保存數據模塊
    if step == 7999:
        saver = tf.compat.v1.train.Saver()
        saver.save(sess, save_path="/home/gaofei/rer/model/leftmodel/Matrixleftbi")





