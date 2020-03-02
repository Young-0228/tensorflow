
# 前向传播

import tensorflow as tf

# 神经网络输入节点为784
INPUT_NODE = 784
# 输出10个数,每个数表示索引号出现的概率
OUTPUT_NODE = 10
# 隐藏层的节点个数
LAYER1_NODE = 500

def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape, stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


# 前向传播
def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2

    return y
