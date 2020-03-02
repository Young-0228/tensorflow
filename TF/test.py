
# 测试程序

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import forward
import backward

TEST_INTERVAL_SECS = 5


def test(mnist):
    # 绘制计算图中的节点
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
        y = forward.forward(x, None)

        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                # 加载ckpt模型
                ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
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


if __name__ == '__main__':
    main()
