#使用TensorFlow实现ResNet

from datetime import datetime
import time
import math
import collections
import tensorflow as tf
slim=tf.contrib.slim

class Block(collections.namedtuple('Block',['scope','unit_fn','args'])):
    'A named tuple describing a ResNet block.'


def subsample(inputs,factor,scope=None):
    '''
    将采样函数.
    '''
    if factor==1:
        return inputs
    else:
        return slim.max_pool2d(inputs,[1,1],stride=factor,scope=scope)

def conv2d_same(inputs,num_outputs,kernel_size,stride,scope=None):
    '''
    Strided 2-D convolution with 'SAME' padding.
    When stride > 1, then we do explicit zero-padding, followed by conv2d with 'VALID' padding.
    Note that
         net = conv2d_same(inputs, num_outputs, 3, stride=stride)
    is equivalent to
         net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
         net = subsample(net, factor=stride)
         
    :param inputs: 输入tensor
    :param num_outputs: 输出通道数
    :param kernel_size: 卷积核尺寸
    :param stride: 步长
    :param scope: Scope
    :return: 卷积输出
    '''

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
    '循环堆叠Blocks'
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
    '定义残差学习单元'
    with tf.variable_scope(scope,'bottleneck_v2',[inputs]) as sc:
        #输入通道数
        depth_in=slim.utils.last_dimension(inputs.get_shape(),min_rank=4)
        #对输入进行预激活
        preact=slim.batch_norm(inputs,activation_fn=tf.nn.relu,scope='preact')
        if depth==depth_in:
            shortcut=subsample(preact,stride,'shortcut')
        else:
            shortcut=slim.conv2d(preact,depth,[1,1],stride=stride,normalizer_fn=None,activation_fn=None,scope='shortcut')
        residual=slim.conv2d(preact,depth_bottleneck,[1,1],stride=1,scope='conv1')
        residual=conv2d_same(residual,depth_bottleneck,3,stride,scope='conv2')
        residual=slim.conv2d(residual,depth,[1,1],stride=1,normalizer_fn=None,activation_fn=None,scope='conv3')
        output=shortcut+residual

        return slim.utils.collect_named_outputs(outputs_collections,sc.name,output)

def resnet_v2(inputs,blocks,num_classes=None,global_pool=True,include_root_block=True,reuse=None,scope=None):
    '''
    生成ResNet v2的主函数
    
    :param inputs: 输入tensor
    :param blocks: Block类列表
    :param num_classes: 分类数
    :param global_pool: 是否加上最后的全局平局池化
    :param include_root_block: 是否加上网络最前面通常使用的7x7卷积和最大池化
    :param reuse: 是否重用
    :param scope: 网络名称
    '''
    with tf.variable_scope(scope,'resnet_v2',[inputs],reuse=reuse) as sc:
        end_points_collection=sc.original_name_scope+'_end_points'
        with slim.arg_scope([slim.conv2d,bottleneck,stack_blocks_dense],outputs_collections=end_points_collection):
            net=inputs
            if include_root_block:
                with slim.arg_scope([slim.conv2d],activation_fn=None,normalizer_fn=None):
                    net=conv2d_same(net,64,7,stride=2,scope='conv2')
                net=slim.max_pool2d(net,[3,3],stride=2,scope='pool1')
            net=stack_blocks_dense(net,blocks)
            net=slim.batch_norm(net,activation_fn=tf.nn.relu,scope='postnorm')
            if global_pool:
                net=tf.reduce_mean(net,[1,2],name='pool5',keep_dims=True)
            if num_classes is not None:
                net=slim.conv2d(net,num_classes,[1,1],activation_fn=None,normalizer_fn=None,scope='logits')
            end_points=slim.utils.convert_collection_to_dict(end_points_collection)
            if num_classes is not None:
                end_points['predictions']=slim.softmax(net,scope='predictions')
            return net,end_points

def resnet_v2_50(inputs,num_classes=None,global_pool=True,reuse=None,scope='resnet_v2_50'):
    blocks=[
        Block('block1',bottleneck,[(256,64,1)]*2+[(256,64,2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024,256, 1)] * 5 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048,512, 1)] * 3)]

    return resnet_v2(inputs,blocks,num_classes,global_pool,include_root_block=True,reuse=reuse,scope=scope)


def resnet_v2_101(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_v2_101'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)]

    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_152(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_v2_152'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)]

    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)

def resnet_v2_200(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_v2_200'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)]

    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)

def time_tensorflow_run(session,target,info_string):
    num_steps_burn_in=10       #预热轮数
    total_duration=0.0
    total_duration_squared=0.0

    for i in range(num_batches+num_steps_burn_in):
        start_time=time.time()
        _=session.run(target)
        duration=time.time()-start_time
        if i>=num_steps_burn_in:
            if not i%10:
                print('%s:step %d,duration=%.3f' % (datetime.now(),i-num_steps_burn_in,duration))
            total_duration+=duration
            total_duration_squared+=duration*duration

    mn=total_duration/num_batches   #平均耗时
    vr=total_duration_squared/num_batches-mn*mn
    sd=math.sqrt(vr)   #标准差

    print('%s:%s across %d steps,%.3f +/- %.3f sec/batch' % (datetime.now(),info_string,num_batches,mn,sd))

batch_size=32
height,width=224,224
inputs=tf.random_uniform((batch_size,height,width,3))
with slim.arg_scope(resnet_arg_scope(is_training=False)):
    net,end_points=resnet_v2_152(inputs,1000)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
#sess.run(net)
#train_writer=tf.summary.FileWriter('/tmp/alog',sess.graph)
num_batches=100
time_tensorflow_run(sess,net,"Forward")