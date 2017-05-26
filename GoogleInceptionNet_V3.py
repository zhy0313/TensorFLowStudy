#使用TensorFlow实现Google Inception Net V3

import tensorflow as tf
slim=tf.contrib.slim
trunc_normal=lambda stddev:tf.truncated_normal_initializer(0.0,stddev)

#生成网络中常用函数的默认参数
def inception_v3_arg_scope(weight_decay=0.00004,stddev=0.1,batch_norm_var_collection='moving_vars'):
    #batch normalization的参数字典
    batch_norm_params={
        'decay':0.9997,
        'epsilon':0.001,
        'updates_collections':tf.GraphKeys.UPDATE_OPS,
        'variables_collections':{
            'beta':None,
            'gamma':None,
            'moving_mean':[batch_norm_var_collection],
            'moving_variance':[batch_norm_var_collectio],
        }
    }

    with slim.arg_scope([slim.conv2d,slim.fully_connected],weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params) as sc:
            return sc

#用于生成函数的卷积部分
def inception_v3_base(inputs,scope=None):
    """
    :param inputs:输入图片数据的tensor 
    :param scope: 包含函数默认参数的环境
    :return: 
    """

    end_points={}
    with tf.variable_scope(scope,'InceptionV3',[inputs]):
        #非Inception Module卷积层
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='VALID'):
            net=slim.conv2d(inputs,32,[3,3],stride=2,scope='Conv2d_1a_3x3')
            net=slim.conv2d(net,32,[3,3],scope='Conv2d_2a_3x3')
            net=slim.conv2d(net,64,[3,3],padding='SAME',scope='Conv2d_2b_3x3')
            net=slim.max_pool2d(net,[3,3],stride=2,scope='MaxPool_3a_3x3')
            net=slim.conv2d(net,80,[1,1],scope='Conv2d_3b_1x1')
            net=slim.conv2d(net,192,[3,3],scope='Conv2d_4a_3x3')
            net=slim.max_pool2d(net,[3,3],stride=2,scope='MaxPool_5a_3x3')

        #Inception Module卷积层
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='SAME'):
            #第一个Inception模组的第一个Inception Module
            with tf.variable_scope('Mixed_5b'):
                with tf.variable_scope('Branch_0'):
                    branch_0=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')

                with tf.variable_scope('Branch_1'):
                    branch_1=slim.conv2d(net,48,[1,1],scope='Conv2d_0a_1x1')
                    branch_1=slim.conv2d(branch_1,64,[5,5],scope='Conv2d_0b_5x5')

                with tf.variable_scope('Branch_2'):
                    branch_2=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0b_3x3')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0c_3x3')

                with tf.variable_scope('Branch_3'):
                    branch_3=slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')
                    branch_3=slim.conv2d(branch_3,32,[1,1],scope='Conv2d_0b_1x1')

                net=tf.concat([branch_0,branch_1,branch_2,branch_3],3)

            #第一个Inception模组的第二个Inception Module
            with tf.variable_scope('Mixed_5c'):
                with tf.variable_scope('Branch_0'):
                    branch_0=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')

                with tf.variable_scope('Branch_1'):
                    branch_1=slim.conv2d(net,48,[1,1],scope='Conv2d_0a_1x1')
                    branch_1=slim.conv2d(branch_1,64,[5,5],scope='Conv2d_0b_5x5')

                with tf.variable_scope('Branch_2'):
                    branch_2=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0b_3x3')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0c_3x3')

                with tf.variable_scope('Branch_3'):
                    branch_3=slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')
                    branch_3=slim.conv2d(branch_3,64,[1,1],scope='Conv2d_0b_1x1')

                net=tf.concat([branch_0,branch_1,branch_2,branch_3],3)

            #第一个Inception模组的第三个Inception Module
            with tf.variable_scope('Mixed_5d'):
                with tf.variable_scope('Branch_0'):
                    branch_0=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')

                with tf.variable_scope('Branch_1'):
                    branch_1=slim.conv2d(net,48,[1,1],scope='Conv2d_0a_1x1')
                    branch_1=slim.conv2d(branch_1,64,[5,5],scope='Conv2d_0b_5x5')

                with tf.variable_scope('Branch_2'):
                    branch_2=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0b_3x3')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0c_3x3')

                with tf.variable_scope('Branch_3'):
                    branch_3=slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')
                    branch_3=slim.conv2d(branch_3,64,[1,1],scope='Conv2d_0b_1x1')

                net=tf.concat([branch_0,branch_1,branch_2,branch_3],3)

            #第二个Inception模组的第一个Inception Module
            with tf.variable_scope('Mixed_6a'):
                with tf.variable_scope('Branch_0'):
                    branch_0=slim.conv2d(net,384,[3,3],stride=2,padding='VALID',scope='Conv2d_0a_3x3')

                with tf.variable_scope('Branch_1'):
                    branch_1=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_3x3')
                    branch_1=slim.conv2d(branch_1,96,[3,3],scope='Conv2d_0b_3x3')
                    branch_1=slim.conv2d(branch_1,96,[3,3],stride=2,padding='VALID',scope='Conv2d_0c_3x3')

                with tf.variable_scope('Branch_2'):
                    branch_2=slim.max_pool2d(net,[3,3],stride=2,padding='VALID',scope='MaxPool_0a_3x3')

                net=tf.concat([branch_0,branch_1,branch_2],3)

            #第二个Inception模组的第二个Inception Module
            with tf.variable_scope('Mixed_6b'):
                with tf.variable_scope('Branch_0'):
                    branch_0=slim.conv2d(net,192,[1,1],scope='Conv2d_0a_1x1')

                with tf.variable_scope('Branch_1'):
                    branch_1=slim.conv2d(net,128,[1,1],scope='Conv2d_0a_1x1')
                    branch_1=slim.conv2d(branch_1,128,[1,7],scope='Conv2d_0b_1x7')
                    branch_1=slim.conv2d(branch_1,192,[7,1],scope='Conv2d_0c_7x1')

                with tf.variable_scope('Branch_2'):
                    branch_2=slim.conv2d(net,128,[1,1],scope='Conv2d_0a_1x1')
                    branch_2=slim.conv2d(branch_2,128,[7,1],scope='Conv2d_0b_7x1')
                    branch_2=slim.conv2d(branch_2,128,[1,7],scope='Conv2d_0c_1x7')
                    branch_2=slim.conv2d(branch_2,128,[7,1],scope='Conv2d_0d_7x1')
                    branch_2=slim.conv2d(branch_2,192,[1,7],scope='Conv2d_0e_1x7')

                with tf.variable_scope('Branch_3'):
                    branch_3=slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')
                    branch_3=slim.conv2d(branch_3,192,[1,1],scope='Conv2d_0b_1x1')

                net=tf.concat([branch_0,branch_1,branch_2,branch_3],3)

            #第二个Inception模组的第三个Inception Module
            with tf.variable_scope('Mixed_6c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')

                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')

                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')

                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')

                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            #第二个Inception模组的第四个Inception Module
            with tf.variable_scope('Mixed_6d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')

                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')

                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')

                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')

                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            #第二个Inception模组的第五个Inception Module
            with tf.variable_scope('Mixed_6e'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')

                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')

                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')

                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')

                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            #将Mixed_6e存储于end_points中，作为辅助分类节点
            end_points['Mixed_6e']=net

            #第三个Inception模组的第一个Inception Module
            with tf.variable_scope('Mixed_7a'):
                with tf.variable_scope('Branch_0'):
                    branch_0=slim.conv2d(net,192,[1,1],scope='Conv2d_0a_1x1')
                    branch_0=slim.conv2d(branch_0,320,[3,3],stride=2,padding='VALID',scope='Conv2d_0b_3x3')

                with tf.variable_scope('Branch_1'):
                    branch_1=slim.conv2d(net,192,[1,1],scope='Conv2d_0a_1x1')
                    branch_1=slim.conv2d(branch_1,192,[1,7],scope='Conv2d_0b_1x7')
                    branch_1=slim.conv2d(branch_1,192,[7,1],scope='Conv2d_0c_7x1')
                    branch_1=slim.conv2d(branch_1,192,[3,3],stride=2,padding='VALID',scope='Conv2d_0d_3x3')

                with tf.variable_scope('Branch_2'):
                    branch_2=slim.max_pool2d(net,[3,3],stride=2,padding='VALID',scope='MaxPool_0a_3x3')

                net = tf.concat([branch_0, branch_1, branch_2], 3)

            #第三个Inception模组的第二个Inception Module
            with tf.variable_scope('Mixed_7b'):
                with tf.variable_scope('Branch_0'):
                    branch_0=slim.conv2d(net,320,[1,1],scope='Conv2d_0a_1x1')

                with tf.variable_scope('Branch_1'):
                    branch_1=slim.conv2d(net,384,[1,1],scope='Conv2d_0a,1x1')
                    branch_1=tf.concat([
                        slim.conv2d(branch_1,384,[1,3],scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1,384,[3,1],scope='Conv2d_0c_3x1')],3)

                with tf.variable_scope('Branch_2'):
                    branch_2=slim.conv2d(net,488,[1,1],scope='Conv2d_0a,1x1')
                    branch_2=slim.conv2d(branch_2,384,[3,3],scope='Conv2d_0b,3x3')
                    branch_2=tf.concat([
                        slim.conv2d(branch_2,384,[1,3],scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2,384,[3,1],scope='Conv2d_0d_3x1')],3)

                with tf.variable_scope('Branch_3'):
                    branch_3=slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')
                    branch_3=slim.conv2d(branch_3,192,[1,1],scope='Conv2d_0b_1x1')

                net=tf.concat([branch_0,branch_1,branch_2,branch_3],3)

            #第三个Inception模组的第三个Inception Module
            with tf.variable_scope('Mixed_7c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')

                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a,1x1')
                    branch_1 = tf.concat([
                        slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0c_3x1')], 3)

                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 488, [1, 1], scope='Conv2d_0a,1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b,3x3')
                    branch_2 = tf.concat([
                        slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)

                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')

                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

    return net,end_points









