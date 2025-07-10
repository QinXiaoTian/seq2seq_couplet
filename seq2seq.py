# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 12:39:40 2025

@author: qinxiaotian
模型代码
"""


import tensorflow as tf
import numpy as np

#import seq2seq_utils
from seq2seq_utils import en_text_arr,de_text_arr,get_batches,get_test_arr,id_to_word

#超参数
seq_length = 30
batch_size = 128
#词表大小
max_vocabs = 5000
#嵌入维度  表示一个汉字，用300个浮点数来表示
embedding_dim = 300
lstm_size = 128
lstm_layer = 2
epochs = 5

train_in_path = 'data/train/in.txt'
train_out_path = 'data/train/out.txt'


#数据准备
en_arr ,en_lengths = en_text_arr(train_in_path,seq_length)
de_arr ,de_lengths = de_text_arr(train_out_path,seq_length)


#数据入口
#placeholder尽量要命名
#为什么是None????
en_arr_ = tf.placeholder(tf.int32,[None,seq_length],name='en_arr')
en_lengths_ = tf.placeholder(tf.int32,[None],name='en_lengths')
de_arr_ = tf.placeholder(tf.int32,[None,seq_length],name='de_arr')
de_lengths_ = tf.placeholder(tf.int32,[None],name='de_lengths')
de_lables_ = tf.placeholder(tf.int32,[None,seq_length],name='de_lables')

#搭建网络
#首先是嵌入层
#当字符是0~4999的时候，乘的是这两个矩阵
en_embedding_matrix = tf.get_variable('en_embedding_matrix',
                                      [max_vocabs,embedding_dim])
de_embedding_matrix = tf.get_variable('de_embedding_matrix',
                                      [max_vocabs,embedding_dim])
#5000个0的向量
#当字符5000的时候，乘的是这个矩阵 当word_to_int得到的数是5000时，乘0
zero_embedding = tf.zeros([1,embedding_dim])#难点
#希望padding和unknow的字符在经过嵌入层以后，得到300个0
en_embedding_matrix = tf.concat([en_embedding_matrix,zero_embedding], axis=0)
de_embedding_matrix = tf.concat([de_embedding_matrix,zero_embedding], axis=0)

en_embedding = tf.nn.embedding_lookup(en_embedding_matrix,en_arr_)
de_embedding = tf.nn.embedding_lookup(de_embedding_matrix,de_arr_)


#接下来是RNN网络的搭建
def lstm_cell():
    return tf.nn.rnn_cell.BasicLSTMCell(lstm_size)

with tf.variable_scope('encoder'):
    en_multi = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(lstm_layer)])
    #en_state作为encoder的输出，输入到encoder中
    en_output,en_state = tf.nn.dynamic_rnn(en_multi,
                                           en_embedding,
                                           dtype=tf.float32,
                                           sequence_length=en_lengths_)
    
with tf.variable_scope('decoder'):
    de_multi = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(lstm_layer)])
    #en_state作为encoder的输出，输入到encoder中
    #de_output作为decoder的输出
    de_output,de_state = tf.nn.dynamic_rnn(de_multi,
                                           de_embedding,
                                           initial_state = en_state,
                                           sequence_length=de_lengths_)



#de_output.shape
#Out[18]: TensorShape([Dimension(None), Dimension(30), Dimension(128)])
#接下来要对de_output进行降维（为什么要降维）
#降维后：TensorShape([Dimension(None), Dimension(128)])
de_output = tf.reshape(de_output,[-1,lstm_size])

#max_vocabs+1是因为在上面，en_embedding_matrix最后加了一个0
#stddev=0.1表示：元素是从均值为0、标准差为0.1的正态分布中随机抽取的
#lstm_size个（行）max_vocabs+1（列）向量
softmax_weights = tf.Variable(tf.random_normal([lstm_size,max_vocabs+1],
                                               stddev=0.1))
softmax_bias = tf.Variable(tf.zeros([max_vocabs+1]))

#训练出来的Y
logit = tf.matmul(de_output, softmax_weights) + softmax_bias

#每个位置要预测出来一个数,只要其第一个数
max_id = tf.argmax(logit, axis=1)[0]

#难点
#由于logit和labels的维度不一致，需要用sparse
loss  = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, 
                                                       labels=tf.reshape(de_lables_,[-1]))
#现在loss的维度是，batch_size*seq_length

sequence_mask = tf.sequence_mask(de_lengths_,maxlen = seq_length,dtype=tf.float32)
sequence_mask = tf.reshape(sequence_mask, [-1])
cost = tf.reduce_mean(loss*sequence_mask)

optimizer = tf.train.AdamOptimizer().minimize(cost)

#保存训练好的模型
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        i = 0
        for batch_en_arr,batch_en_lengths,batch_de_arr,batch_de_lengths,batch_de_labels in get_batches(en_arr, en_lengths, de_arr, de_lengths, batch_size):
            feed = {en_arr_:batch_de_arr,
                    en_lengths_:batch_en_lengths,
                    de_arr_:batch_de_arr,
                    de_lengths_:batch_de_lengths,
                    de_lables_:batch_de_labels}
            i += 1
            _,cost_ = sess.run([optimizer,cost],feed_dict = feed)
            #每训练100次，打印损失
            if i %100 == 0:
                print('Epoch {},iteration {}:cost={}'.format(epoch, i,cost_))
            #每训练500次，保存训练数据
            if i %100 == 0:
                saver.save(sess,'new_models/')




#使用已经训练好的网络
test = '春眠不觉晓'
test_en_arr,test_en_length = get_test_arr(test,seq_length)

de_start =['<s>']
test_de_arr ,test_de_length= get_test_arr(de_start,seq_length)

#接下来，我们就要把en_state的值算出来

sess = tf.Session()
#checkpoint = tf.train.latest_checkpoint('new_models/')
#saver.recover_last_checkpoints(sess, checkpoint)
saver.restore(sess, 'new_models/')

feed = {en_arr_:test_en_arr,
        en_lengths_:test_en_length}

en_state_ = sess.run(en_state,feed_dict=feed)
print_arr = []
for i in range(seq_length):
    feed_de = {de_arr_:test_de_arr,
               de_lengths_:test_de_length,
               en_state:en_state_}

    max_id_,en_state = sess.run([max_id,de_state],feed_dict=feed_de)
    word = id_to_word(max_id_)
    test_de_arr ,test_de_length= get_test_arr([word],seq_length)
    if(word == '</s>'):
        break
    print_arr.append(word)
    
print(''.join(print_arr))













