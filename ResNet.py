#使用TensorFlow实现ResNet

import collections
import tensorflow as tf
slim=tf.contrib.slim

class Block(collections.namedtuple('Block',['scope','unit_fn','args'])):
    'A named tuple describing a ResNet block.'

#降采样函数
def subsample(inputs,factor,scope=None):
    if factor==1:
        return inputs
    else:
        return slim.max_pool2d(inputs,[1,1],stride=factor,scope=scope)

#卷积函数
def conv2d_same(inputs,num_outputs,kernel_size,stride,scope=None):
    if stride==1:
        return slim.conv2d(inputs,num_outputs,kernel_size,stride=1,padding='SAME',scope=scope)
    else:
        pad_total=kernel_size-1
        pad_beg=pad_total//2
        pad_end=pad_total-pad_beg
        inputs=tf.pad(inputs,[[0,0],[pad_beg,pad_end],[pad_beg,pad_end],[0,0]])
        return slim.conv2d(inputs,num_outputs,kernel_size,stride=stride,padding='VALID',scope=scope)

@slim.add_arg_scope
def stack_blocks_dense(net,blocks,outputs_collections=None):
    for block in blocks:
        with tf.variable_scope(block.scope,'block',[net]) as sc:
            for i,unit in enumerate(block.args):
                with tf.variable_scope('unit_%d'%(i+1),values=[net]):
                    unit_depth,unit_depth_bottleneck,unit_stride=unit
                    net=block.unit_fn(net,depth=unit_depth,depth_bottleneck=unit_depth_bottleneck,stride=unit_stride)
            net=slim.utils.collect_named_outputs(outputs_collections,sc.name,net)
    return net

def resnet_arg_scope(is_training=True,weight_decay=0.0001,batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,batch_norm_scale=True):
    batch_norm_params={
        'is_training':is_training,
        'decay':batch_norm_decay,
        'epsilon':batch_norm_epsilon,
        'scale':batch_norm_scale,
        'updates_collections':tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=slim.variance_scaling_initializer(),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm],**batch_norm_params):
            with slim.arg_scope([slim.max_pool2d],padding='SAME') as arg_sc:
                return arg_sc

@slim.add_arg_scope
def bottleneck(inputs,depth,depth_bottleneck,stride,outputs_collections=None,scope=None):
    with tf.variable_scope(scope,'bottleneck_v2',[inputs]) as sc:
        depth_in=slim.utils.last_dimension(inputs.get_shape(),min_rank=4)
        preact=slim.batch_norm(inputs,activation_fn=tf.nn.relu,scope='preact')
        if depth==depth_in:
            shortcut=subsample(inputs,stride,'shortcut')
        else:
            shortcut=slim.conv2d(preact,depth,[1,1],stride=stride,normalizer_fn=None,activation_fn=None,scope='shortcut')
        residual=slim.conv2d(preact,depth_bottleneck,[1,1],stride=1,scope='conv1')
        residual=conv2d_same(residual,depth_bottleneck,3,stride,scope='conv2')
        residual=slim.conv2d(residual,depth,[1,1],stride=1,normalizer_fn=None,activation_fn=None,scope='conv3')
        output=shortcut+residual

        return slim.utils.collect_named_outputs(outputs_collections,sc.name,output)



