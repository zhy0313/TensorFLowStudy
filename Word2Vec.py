#使用TensorFlow实现Word2Vec的训练

import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf

url='http://mattmahoney.net/dc/'

def maybe_download(filename,expected_bytes):
    '下载数据'
    if not os.path.exists(filename):
        filename,_=urllib.request.urlretrieve(url+filename,filename)
    statinfo=os.stat(filename)
    if statinfo.st_size==expected_bytes:
        print('Found and verified',filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify'+filename+'. Can you get to it with a browser?'
        )
    return filename

filename=maybe_download('text8.zip',31344016)

def read_data(filename):
    '解压数据，并将数据转为单词列表'
    with zipfile.ZipFile(filename) as f:
        data=tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words=read_data(filename)
print('Data size',len(words))

vocabulary_size=50000   #词汇表尺寸，top50000频数的单词

def build_dataset(words):
    '统计单词频数，创建top50000词汇表，对单词进行编号'
    count=[['UNK',-1]] #频数统计，top50000
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))

    dictionary=dict() #top50000词汇字典
    for word,_ in count:
        dictionary[word]=len(dictionary)

    data=list() #编码
    unk_count=0
    for word in words:
        if word in dictionary:
            index=dictionary[word]
        else:
            index=0
            unk_count+=1
        data.append(index)
    count[0][1]=unk_count
    reverse_dictionary=dict(zip(dictionary.values(),dictionary.keys()))

    return data,count,dictionary,reverse_dictionary

data,count,dictionary,reverse_dictionary=build_dataset(words)

del words
#print('Most common words (+UNK)',count[:5])
#print('Sample data',data[:10],[reverse_dictionary[i] for i in data[:10]])

data_index=0

def generate_batch(batch_size,num_skips,skip_window):
    '''
    生成训练样本
    :param batch_size: batch大小 
    :param num_skips: 对每个单词生成的样本数. num_skips<=2*ship_window,batch_size%num_skips==0
    :param skip_window: 单词最远可以联系的距离
    :return: 用于训练的batch数据
    '''

    global data_index
    assert batch_size % num_skips == 0
    assert num_skips<=2*skip_window
    batch=np.ndarray(shape=(batch_size),dtype=np.int32)
    labels=np.ndarray(shape=(batch_size,1),dtype=np.int32)
    span=2*skip_window+1

    buffer=collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index=(data_index+1)%len(data)

    for i in range(batch_size//num_skips):
        '每次循环对一个目标单词生成样本'
        target=skip_window
        targets_to_avoid=[skip_window]

        for j in range(num_skips):
            while target in targets_to_avoid:
                target=random.randint(0,span-1)
            targets_to_avoid.append(target)
            batch[i*num_skips+j]=buffer[skip_window]
            labels[i*num_skips+j,0]=buffer[target]

        buffer.append(data[data_index])
        data_index=(data_index+1)%len(data)
    return batch,labels

#batch,labels=generate_batch(batch_size=8,num_skips=2,skip_window=1)
#for i in range(8):
#    print(batch[i],reverse_dictionary[batch[i]],'->',labels[i,0],reverse_dictionary[labels[i,0]])

batch_size=128
embedding_size=128  #单词转为稠密向量的维度
skip_window=1
num_skips=2

valid_size=16  #验证单词数
valid_window=100  #验证单词只从频数最高的100个单词中抽取
valid_examples=np.random.choice(valid_window,valid_size,replace=False)  #抽取验证单词
num_sampled=64   #噪声单词数量

