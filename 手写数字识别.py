from PIL import Image
import numpy as np
import time
import tensorflow as tf
# old_v = tf.logging.get_verbosity()
# tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
import os

# # tf.get_collection('')  从集合中取全部变量，生成一个列表
# # tf.add_n('')  列表内对应元素相加
# # tf.cast(x,dtype)  把x转化dtype类型
# # tf.argmax(x,axis)  返回最大值所在的索引号 如：tf.argmax([1,0,0],1),返回0
# # os.path.join('home','name')  返回home/name
# # 字符串.spilt()  按指定拆分符对字符串切片，返回分割后的列表
# # with tf.Graph().as_default as g  其内定义的节点在计算图g中

# 神经网络输入节点为784
INPUT_NODE = 784
# 输出10个数,每个数表示索引号出现的概率
OUTPUT_NODE = 10
# 隐藏层的节点个数
LAYER1_NODE = 500
# 一次训练所选取的样本数
BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model"
TEST_INTERVAL_SECS = 5
# mnist = input_data.read_data_sets('./model/', one_hot=True) # 自动加载数据集
# # tf.logging.set_verbosity(old_v)
# # 训练集所含有的样本数
# print('train data size:',mnist.train.num_examples)
# # 验证集所含有的样本数
# print('validation data size:', mnist.validation.num_examples)
# # 测试集所含有的样本数
# print('test data size:',mnist.test.num_examples)
# # 查看训练集中指定编号的标签 第0张图的标签
# print(mnist.train.labels[54999])
# # # 查看训练集中指定编号的图片 第0张图片的784个像素点
# print(mnist.train.images[54999])

def get_weight(shape,regularizer):
	w = tf.Variable(tf.random_normal(shape,stddev=0.1))
	if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

def get_bias(shape):  
    b = tf.Variable(tf.zeros(shape)) 
    return b
	
# 前向传播
def forward(x, regularizer):
	
	w1 = get_weight([INPUT_NODE,LAYER1_NODE], regularizer)	
	b1 = get_bias([LAYER1_NODE])
	y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

	w2 = get_weight([LAYER1_NODE,OUTPUT_NODE], regularizer)
	b2 = get_bias([OUTPUT_NODE])
	y = tf.matmul(y1, w2) + b2 
	
	return y

#反向传播 
def backward(minist):
	x = tf.placeholder(tf.float32,[None,INPUT_NODE])
	y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE])
	y = forward(x,REGULARIZER)
	global_step = tf.Variable(0,trainable=False)

	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
	cem = tf.reduce_mean(ce)
	loss = cem+tf.add_n(tf.get_collection('losses'))

	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,
		global_step,
		minist.train.num_examples / BATCH_SIZE,
		LEARNING_RATE_DECAY,
		staircase=True
	)

	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

	ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
	ema_op = ema.apply(tf.trainable_variables())
	with tf.control_dependencies([train_step,ema_op]):
		train_op = tf.no_op(name='train')
    # 保存模型
	saver = tf.train.Saver()  # 实例化saver对象
    
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
        
        # 断点续训
        # ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(sess,ckpt.model_checkpoint_path)
        

		for i in range(STEPS):
			xs, ys = minist.train.next_batch(BATCH_SIZE)
			_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
			if i % 1000 == 0:
				print("After %d training step(s), loss on training batch is %g." % (step-1, loss_value))
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    # tf.logging.set_verbosity(old_v)
    backward(mnist)

# if __name__ == '__main__':
#     main()



def test(mnist):
    # 绘制计算图中的节点
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE])
        y = forward(x, None)

        ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
		
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))		

        while True:
            with tf.Session() as sess:
                # 加载ckpt模型
                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                # 如果已有ckpt模型则恢复
                if ckpt and ckpt.model_checkpoint_path:
                    # 恢复会话
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 恢复轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    # 计算准确率
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    # 打印提示
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                # 如果没有模型
                else:
                    # 给出提示
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)

def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    test(mnist)

# if __name__ == '__main__':
#     main()


def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32,[None,INPUT_NODE])
        y = forward(x,None)
        preValue = tf.argmax(y,1)

        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                preValue = sess.run(preValue,feed_dict={x:testPicArr})
                return preValue

            else:
                print('No checkpoint file found')
                return -1

def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((28,28),Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))
    threshold = 50
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
    nm_arr = im_arr.reshape([1,784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr,1.0/255.0)
    return img_ready

def application():
    testNum = int(input("input the number of test pictures:"))
    for i in range(testNum):
        testPic = input("the path of test picture:")
        # 先对手写数字图片testpic做预处理
        testPicArr = pre_pic(testPic)
        # 当图片符合神经网络输入要求后再喂给复现的神经网络模型，输出预测值
        preValue = restore_model(testPicArr)
        print('The prediction number is:',preValue)

def main():
    application()

if __name__ == "__main__":
    main()    