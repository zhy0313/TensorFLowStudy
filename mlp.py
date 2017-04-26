#实现多层感知机

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
sess=tf.InteractiveSession()

in_units=784     #输入节点数
h1_units=300     #隐含层节点数

#权重初始化
W1=tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))  #截断的正态分布，标准差为0.1
b1=tf.Variable(tf.zeros([h1_units]))
W2=tf.Variable(tf.zeros([h1_units,10]))
b2=tf.Variable(tf.zeros([10]))

x=tf.placeholder(tf.float32,[None,in_units])
keep_prob=tf.placeholder(tf.float32)             #Dropout比率

#模型结构
hidden1=tf.nn.relu(tf.matmul(x,W1)+b1)
hidden1_drop=tf.nn.dropout(hidden1,keep_prob)    #Dropout
y=tf.nn.softmax(tf.matmul(hidden1_drop,W2)+b2)

y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))  #损失函数，交叉信息熵
train_step=tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)     #使用Adagrad优化器进行优化，学习速率0.3

tf.global_variables_initializer().run()

#迭代训练
#使用3000个batch,每个batch包含100条样本
for i in range(3000):
    batch_x,batch_ys=mnist.train.next_batch(100)
    train_step.run({x:batch_x,y_:batch_ys,keep_prob:0.75})

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuary=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuary.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))



