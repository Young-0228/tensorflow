# import tensorflow as tf

# a = tf.constant([1.0,2.0])
# b = tf.constant([3.0,4.0])
# result = a+b
# print(result)

# Tensor("add:0", shape=(2,), dtype=float32)
# add:节点名   0:第0个输出  shape:维度，一维数组长度2  dtype:数据类型 浮点型

# # 生产一批零件，输入体积和重量，通过NN后输出一个值
# # 两层简单神经网络(全连接)
# import tensorflow as tf
# # 定义输入和参数
# x = tf.constant([[0.7,0.5]])
# w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
# w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

# # 定义向前传播过程
# a = tf.matmul(x,w1)
# y = tf.matmul(a,w2)

# # 用会话计算结果
# with tf.Session() as sess:
#     # 变量初始化，在sess.run函数中用tf.global_variables_initializer()
#     init_op = tf.global_variables_initializer()
#     # 计算图节点运算：在sess.run函数中写入待运算的节点
#     sess.run(init_op)
#     print(sess.run(y))




# # 生产一批零件，输入体积和重量，通过NN后输出一个值
# # 两层简单神经网络(全连接)
# import tensorflow as tf
# # 定义输入和参数
# # 用placeholder实现输入定义(sess.run中喂一组数据)
# # 用tf.placeholder占位,在sess.run函数中用fed_dict喂数据
# x = tf.placeholder(tf.float32, shape=(1, 2))
# w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
# w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

# # 定义向前传播过程
# a = tf.matmul(x,w1)
# y = tf.matmul(a,w2)

# # 用会话计算结果
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     print(sess.run(y, feed_dict={x:[[0.7,0.5]]}))



# import tensorflow as tf
# # 定义输入和参数
# x = tf.placeholder(tf.float32,shape=(None,2))
# w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
# w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
# # 定义向前传播过程
# a = tf.matmul(x,w1)
# y = tf.matmul(a,w2)
# # 调用会话计算结果
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     print(sess.run(y,feed_dict={x:[[0.7,0.5],[0.2,0.3],[0.3,0.4],[0.4,0.5]]}))
#     print('w1:')
#     print(sess.run(w1))
#     print('w2:')
#     print(sess.run(w2))


# import tensorflow as tf
# import numpy as np
# # 一次训练所选取的样本数
# BATCH_SIZE = 8
# seed = 23455
# # 基于seed产生随机数
# rng =np.random.RandomState(seed)
# # 随机数返回32行2列的矩阵，表示32组体积和重量 作为输入数据集
# X = rng.rand(32,2)
# # 从X这个32行2列的矩阵中取出一行，判断如果和小于1给Y赋值1，如果和不小于1给Y赋值0
# # 作为输入数据集的标签(正确答案)
# Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]
# print(X)
# print(Y)
# # 定义神经网络的输入、参数和输出，定义前向传播过程
# x = tf.placeholder(tf.float32,shape=(None,2))
# y_ = tf.placeholder(tf.float32,shape=(None,1))
# # tf.random_normal()函数用于从服从指定正太分布的数值中取出指定个数的值。
# # stddev: 正态分布的标准差
# # seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
# w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
# w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
# # tf.matmul()将矩阵a乘以矩阵b，生成a * b
# a = tf.matmul(x,w1)
# y = tf.matmul(a,w2)
# # 定义损失函数及反向传播方法
# # 均方误差
# loss = tf.reduce_mean(tf.square(y-y_))
# # 优化方法：梯度下降实现训练过程  学习率0.001
# train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# # tf.train.GradientDescentOptimizer()使用随机梯度下降算法
# # train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# # tf.train.AdamOptimizer()是利用自适应学习率的优化算法
# # train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# # 生成会话，训练STEPS轮
# with tf.Session() as sess:
#     # 变量初始化，在sess.run函数中用tf.global_variables_initializer()
#     init_op = tf.global_variables_initializer()
#     # 计算图节点运算：在sess.run函数中写入待运算的节点
#     sess.run(init_op)
#     print('w1:')
#     print(sess.run(w1))
#     print('w2:')
#     print(sess.run(w2))

#     # 训练模型
#     STEPS = 3000
#     for i in range(STEPS):
#         start = (i*BATCH_SIZE) % 32
#         end = start + BATCH_SIZE
#         sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
#         if i % 500 == 0:
#             total_loss = sess.run(loss,feed_dict={x:X,y_:Y})
#             print("After %d training step(s),loss on all data is %f" % (i,total_loss))
#     # 输出训练后的参数取值
#     print('w1:')
#     print(sess.run(w1))
#     print('w2:')
#     print(sess.run(w2))


# import tensorflow as tf
# import numpy as np
# # 一次训练所选取的样本数
# BATCH_SIZE = 8
# seed = 23455
# # 基于seed产生随机数
# rng = np.random.RandomState(seed)
# # 生成32行2列的矩阵，表示32组体积和重量
# X = rng.rand(32,2)
# # 从32行中取出一行，体积和重量相加，如果小于1就给Y赋值1，如果大于一，就给Y赋值0
# # 作为输入数据集的标签
# Y = [[int(x0 +x1) < 1] for (x0,x1) in X]
# print(X)
# print(Y)
# # 定义神经网络的输入、参数、输出及前向传播
# x = tf.placeholder(tf.float32,shape=(None,2))
# y_ = tf.placeholder(tf.float32,shape=(None,1))
# w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
# w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
# a = tf.matmul(x,w1)
# y = tf.matmul(a,w2)
# # 定义损失函数及反向传播
# loss = tf.reduce_mean(tf.square(y-y_))
# # 使用梯度下降算法
# train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# # 生成会话，训练step轮
# with tf.Session() as sess:
#     # 变量初始化
#     init_op = tf.global_variables_initializer()
#     # 计算图节点运算
#     sess.run(init_op)
#     print("w1:",sess.run(w1))
#     print("w2:",sess.run(w2))

#     STEPS = 3000
#     for i in range(STEPS):
#         start = (i*BATCH_SIZE) % 32
#         end = BATCH_SIZE + start
#         sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
#         if i % 500 == 0:
#             total_loss = sess.run(loss,feed_dict={x:X,y_:Y})
#             print("After %d training step(s),loss on all data is %f" % (i,total_loss))
#     print("w1:",sess.run(w1))
#     print("w2:",sess.run(w2))
    



'''

# 预测酸奶日销量y，x1、x2是影响y的因素
import tensorflow as tf
import numpy as np
# 一次训练所选取的样本数
BATCH_SIZE = 8
SEED = 23455

# 基于seed产生随机数
rdm = np.random.RandomState(SEED)
X = rdm.rand(32,2)
# rdm.rand()生成0~1的前闭后开区间的随机数        随机噪声 构建标准答案
Y_ = [[x1+x2 + (rdm.rand()/10.0-0.05)] for (x1,x2) in X]

# 定义神经网络的输入、参数和输出，定义前向传播过程
x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))
w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y = tf.matmul(x,w1)

# 定义损失函数及反向传播方法
# 定义损失函数为MSE，反向传播方法为梯度下降
loss_mse = tf.reduce_mean(tf.square(y_-y))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)

# 生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 20000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = (i*BATCH_SIZE) % 32 + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 500 == 0:
            print("After %d training steps, w1 is: " % (i))
            print(sess.run(w1))
    print("Final w1 is: ",sess.run(w1))
# 拟合的结果是销量y = 0.98 * x1 + 1.02 * x2


# 自定义损失函数
# 预测酸奶日销量y，x1、x2是影响y的因素
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455
COST = 1
PROFIT = 9

rdm = np.random.RandomState(SEED)
X = rdm.rand(32,2)
# rdm.rand()生成0~1的前闭后开区间的随机数        随机噪声 构建标准答案
Y_ = [[x1+x2 + (rdm.rand()/10.0-0.05)] for (x1,x2) in X]

# 定义神经网络的输入、参数和输出，定义前向传播过程
x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))
w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y = tf.matmul(x,w1)

# 定义损失函数及反向传播方法
# 定义损失函数使得预测少了的损失大,于是模型应该偏向多的方向预测。
# tf.where(tf.greater(y,y_))判断y与y_的大小，若y大，输出(y - y_) * COST
# tf.reduce_sum 所有损失求和
loss_mse = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * COST, (y_-y) * PROFIT))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)

# 生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 20000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = (i*BATCH_SIZE) % 32 + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 500 == 0:
            print("After %d training steps, w1 is: " % (i))
            print(sess.run(w1))
    print("Final w1 is: ",sess.run(w1))
# 拟合的结果是销量y = 0.98 * x1 + 1.02 * x2

'''

# import tensorflow as tf
# # 定义待优化参数w初始值赋5
# w = tf.Variable(tf.constant(5,dtype=tf.float32))
# # 定义损失函数loss
# loss = tf.square(w+1)
# # 定义反向传播方法
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     for i in range(40):
#         sess.run(train_step)
#         w_val = sess.run(w)
#         loss_val = sess.run(loss)
#         print("After %s steps: w is %f loss is %f" % (i,w_val,loss_val))


# import tensorflow as tf
# # 1、定义变量及滑动平均类
# # 定义一个32位浮点变量，初始值为0.0，这个代码就是不断更新w1参数，优化w1参数，滑动平均做了个w1的影子
# w1 = tf.Variable(0,dtype=tf.float32)
# # 定义num_updates(NN的迭代轮数)，初始值为0，不可被优化，这个参数不训练
# global_step = tf.Variable(0,trainable=False)
# # 实例化滑动平均类，给删减率为0.99，当前轮数global_step
# MOVING_AVERAGE_DECAY = 0.99
# ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
# # ema.apply后的括号里是更新列表，每次运行sess.run(ema_op)时，对更新列表中的元素求滑动平均值
# # 在实际应用中会使用tf.Variable_variables()自动将所有待训练的参数汇总为列表
# # ema_op = ema.apply([w1])
# ema_op = ema.apply(tf.trainable_variables())
# # 2、查看不同迭代中变量取值变化
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     # 用ema.average(w1)获取w1滑动平均值(要运行多个节点，作为列表中的元素列出，写在sess.run中)
#     print(sess.run([w1,ema.average(w1)]))

#     # 参数w1的值赋为1
#     sess.run(tf.assign(w1,1))
#     sess.run(ema_op)
#     print(sess.run([w1,ema.average(w1)]))

#     # 更新step和w1的值，模拟出100轮迭代后，参数w1变为10
#     sess.run(tf.assign(global_step,100))
#     sess.run(tf.assign(w1,10))
#     sess.run(ema_op)
#     print(sess.run([w1,ema.average(w1)]))

#     # 每次sess.run会更新一次w1的滑动平均值
#     sess.run(ema_op)
#     print(sess.run([w1,ema.average(w1)]))
#     sess.run(ema_op)
#     print(sess.run([w1,ema.average(w1)]))
#     sess.run(ema_op)
#     print(sess.run([w1,ema.average(w1)]))
#     sess.run(ema_op)
#     print(sess.run([w1,ema.average(w1)]))
#     sess.run(ema_op)
#     print(sess.run([w1,ema.average(w1)]))
#     sess.run(ema_op)
#     print(sess.run([w1,ema.average(w1)]))



# import numpy as np
# import matplotlib.pyplot as plts
# import tensorflow as tf

# seed = 2 
# def generateds():
# 	#基于seed产生随机数
# 	rdm = np.random.RandomState(seed)
# 	#随机数返回300行2列的矩阵，表示300组坐标点（x0,x1）作为输入数据集
# 	X = rdm.randn(300,2)
# 	#从X这个300行2列的矩阵中取出一行,判断如果两个坐标的平方和小于2，给Y赋值1，其余赋值0
# 	#作为输入数据集的标签（正确答案）
# 	Y_ = [int(x0*x0 + x1*x1 <2) for (x0,x1) in X]
# 	#遍历Y中的每个元素，1赋值'red'其余赋值'blue'，这样可视化显示时人可以直观区分
# 	Y_c = [['red' if y else 'blue'] for y in Y_]
# 	#对数据集X和标签Y进行形状整理，第一个元素为-1表示跟随第二列计算，第二个元素表示多少列，可见X为两列，Y为1列
# 	X = np.vstack(X).reshape(-1,2)
# 	Y_ = np.vstack(Y_).reshape(-1,1)
	
# 	return X, Y_, Y_c

# def get_weight(shape, regularizer):
# 	w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
# 	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
# 	return w

# def get_bias(shape):  
#     b = tf.Variable(tf.constant(0.01, shape=shape)) 
#     return b
	
# def forward(x, regularizer):
	
# 	w1 = get_weight([2,11], regularizer)	
# 	b1 = get_bias([11])
# 	y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

# 	w2 = get_weight([11,1], regularizer)
# 	b2 = get_bias([1])
# 	y = tf.matmul(y1, w2) + b2 
	
# 	return y

# STEPS = 40000
# BATCH_SIZE = 30 
# LEARNING_RATE_BASE = 0.001
# LEARNING_RATE_DECAY = 0.999
# REGULARIZER = 0.01

# def backward():
# 	x = tf.placeholder(tf.float32, shape=(None, 2))
# 	y_ = tf.placeholder(tf.float32, shape=(None, 1))

# 	X, Y_, Y_c = generateds()

# 	y = forward(x, REGULARIZER)
	
# 	global_step = tf.Variable(0,trainable=False)	

# 	learning_rate = tf.train.exponential_decay(
# 		LEARNING_RATE_BASE,
# 		global_step,
# 		300/BATCH_SIZE,
# 		LEARNING_RATE_DECAY,
# 		staircase=True)


# 	#定义损失函数
# 	loss_mse = tf.reduce_mean(tf.square(y-y_))
# 	loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))
	
# 	#定义反向传播方法：包含正则化
# 	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)

# 	with tf.Session() as sess:
# 		init_op = tf.global_variables_initializer()
# 		sess.run(init_op)
# 		for i in range(STEPS):
# 			start = (i*BATCH_SIZE) % 300
# 			end = start + BATCH_SIZE
# 			sess.run(train_step, feed_dict={x: X[start:end], y_:Y_[start:end]})
# 			if i % 2000 == 0:
# 				loss_v = sess.run(loss_total, feed_dict={x:X,y_:Y_})
# 				print("After %d steps, loss is: %f" %(i, loss_v))

# 		xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
# 		grid = np.c_[xx.ravel(), yy.ravel()]
# 		probs = sess.run(y, feed_dict={x:grid})
# 		probs = probs.reshape(xx.shape)
	
# 	plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c)) 
# 	plt.contour(xx, yy, probs, levels=[.5])
# 	plt.show()
	
# if __name__=='__main__':
# 	backward()


from PIL import Image
import numpy as np
import time
import tensorflow as tf
# old_v = tf.logging.get_verbosity()
# tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
import os

# 神经网络输入节点为784
INPUT_NODE = 784
# 输出10个数,每个数表示索引号出现的概率
OUTPUT_NODE = 10
# 隐藏层的节点个数
LAYER1_NODE = 500
BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 20000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model"
TEST_INTERVAL_SECS = 5

def get_weight(shape,regularizer):
	w = tf.Variable(tf.random_normal(shape,stddev=0.1))
	if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

def get_bias(shape):  
    b = tf.Variable(tf.zeros(shape)) 
    return b
	
def forward(x, regularizer):
	
	w1 = get_weight([INPUT_NODE,LAYER1_NODE], regularizer)	
	b1 = get_bias([LAYER1_NODE])
	y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

	w2 = get_weight([LAYER1_NODE,OUTPUT_NODE], regularizer)
	b2 = get_bias([OUTPUT_NODE])
	y = tf.matmul(y1, w2) + b2 
	
	return y

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
            # saver.restore(sess,ckpt.model_checkpoint_path)
        

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
                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)

def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    test(mnist)

if __name__ == '__main__':
    main()


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
    testNum = input("input the number of test pictures:")
    for i in range(testNum):
        testPic = raw_input("the path of test picture:")
        testPicArr = pre_pic(testPic)
        preValue = restore_model(testPicArr)
        print('The prediction number is:',preValue)

def main():
    application()

if __name__ == "__main__":
    main()    