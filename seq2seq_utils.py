# -*- coding: utf-8 -*-
"""
训练对联的模型
"""
import numpy as np


vocabs_path = 'data/vocabs'
train_in_path = 'data/train/in.txt'
train_out_path = 'data/train/out.txt'
#字典表
vocabs = []
#序列的长度，这里指的是对联的单词个数如：新 居 落 成 创 业 始
#不足30个字，要进行填充
seq_length = 30
#因为总的字表大小是9130，越到后面的字都是生僻字
#为了减小计算开销，只用前5000个字
max_vocabs = 5000



#使用数据集中已经有的单词表vocabs
#因为vocabs里存的是中文，所以要加上encoding='utf-8'
with open(vocabs_path,encoding='utf-8') as f:
    #把所有行中的\n去掉
    for line in f:
        vocabs.append(line.strip())
    
vocabs = vocabs[:max_vocabs]

#单词转换成数字
word_to_int = {word:i for i,word in enumerate(vocabs)}

#数字转换成单词
int_to_word = dict(enumerate(vocabs))

    
#上联处理
def en_text_arr(train_in_path,seq_length):
    #我们要一个en_arr.shape=(None,seq_length)
    #每一条对联的长度en_length=(None)
    #总共是四个元素：开始、结束、unknow、padding
    with open(train_in_path,encoding='utf-8') as f:
        en_arr = []
        en_lengths = [] 
        for line in f:
            #未使用split()之前：'新 居 落 成 创 业 始 \n'
            #使用split()之后：['新', '居', '落', '成', '创', '业', '始']
            words = line.split()
            arr = []
            #当前上联有多少个字
            len_word = len(words)
            #一般的写法是这个，将每一行的字符转为数字
            #arr = [word_to_int[word] for word in words]
            #但是在中文字表中，我们只取了一部分数据，如果有生僻字，就会出错
            #首先把每个字符在字典表里找到对应的序号，如果找不到，添加5000
            #这里把unknow解决了，unknow也指生僻字
            for word in words:
                #非生僻字
                if word in word_to_int:
                    arr.append(word_to_int[word])
                #生僻字，记为5000
                else:
                    arr.append(max_vocabs)
                    
            #接下来就是padding
            #如果上联的长度，小于最大长度seq_length
            if(len_word<seq_length):
                #padding也是用5000代替
                #小于则填充，用5000代替
                #如[30,368,91,63,540,117,732,5000,5000,
                #5000,5000,5000,5000,5000,5000,5000,5000,
                #5000,5000,5000,5000,5000,5000,5000,5000,5000,5000,5000,5000,5000]
                arr+=[max_vocabs]*(seq_length-len_word)
                en_lengths.append(len_word)
            else:
                #超出部分，则截掉
                arr=arr[:seq_length]
                en_lengths.append(seq_length)
                
            en_arr.append(arr)
            
        return np.array(en_arr),np.array(en_lengths)
#下联处理
#输入和输出是错位的
#解码输入de_seq:<s> x1 x2 x3 x4 x5
#解码输出de_label:x1 x2 x3 x4 x5 </s>
def de_text_arr(train_out_path,seq_length):
    #我们要一个de_arr.shape=(None,seq_length)
    #每一条对联的长度de_length=(None)
    #总共是四个元素：开始、结束、unknow、padding
    with open(train_out_path,encoding='utf-8') as f:
        de_arr = []
        de_lengths = [] 
        for line in f:
            #未使用split()之前：'新 居 落 成 创 业 始 \n'
            #使用split()之后：['新', '居', '落', '成', '创', '业', '始']
            words = line.split()
            words = ['<s>'] + words + ['</s>']
            arr = []
            #当前下联有多少个字
            len_word = len(words)
            #一般的写法是这个，将每一行的字符转为数字
            #arr = [word_to_int[word] for word in words]
            #但是在中文字表中，我们只取了一部分数据，如果有生僻字，就会出错
            #首先把每个字符在字典表里找到对应的序号，如果找不到，添加5000
            #这里把unknow解决了，unknow也指生僻字
            for word in words:
                #非生僻字
                if word in word_to_int:
                    arr.append(word_to_int[word])
                #生僻字，记为5000
                else:
                    arr.append(max_vocabs)
                    
            #接下来就是padding
            #如果下联的长度，小于最大长度seq_length
            #seq_length+1操作是因为多了</s>
            if(len_word<seq_length+1):
                #padding也是用5000代替
                #小于则填充，用5000代替
                #如[30,368,91,63,540,117,732,5000,5000,
                #5000,5000,5000,5000,5000,5000,5000,5000,
                #5000,5000,5000,5000,5000,5000,5000,5000,5000,5000,5000,5000,5000]
                arr+=[max_vocabs]*(seq_length+1-len_word)
                de_lengths.append(len_word)
            else:
                #超出部分，则截掉
                arr=arr[:seq_length+1]
                de_lengths.append(seq_length)
                
            de_arr.append(arr)
            
        return np.array(de_arr),np.array(de_lengths)

def get_batches(en_arr,en_lengths,de_arr,de_lengths,batch_size):
    #训练一次的数据大小，为方便训练，最后一位的余数不要
    num_batches = en_arr.shape[0]//batch_size
    #训练数据最后一位的索引，本训练数据集大小为：770491
    end_index = num_batches*batch_size
    en_arr = en_arr[:end_index]
    en_lengths = en_lengths[:end_index]
    de_arr = de_arr[:end_index]
    de_lengths = de_lengths[:end_index]
    #从0到en_index遍历，每一次长度是batch_size
    for i in range(0,end_index,batch_size):
        #每一批的数据设置
        batch_en_arr = en_arr[i:i+batch_size]
        batch_en_lengths = en_lengths[i:i+batch_size]
        #:-1表示取的数，从第1个到倒数第二个值
        #如a为array([[1, 2, 3],
        #          [4, 5, 6],
        #          [7, 8, 9]])
        #a[0:2,-1]输出：array([[1, 2],
        #                     [4, 5]])
        batch_de_arr = de_arr[i:i+batch_size,:-1]#解码器的输入需要是seq_length的长度，应该是30
        batch_de_lengths = de_lengths[i:i+batch_size]
        #de_labels与de_arr要错位训练
        batch_de_labels = de_arr[i:i+batch_size,1:]#标签也是seq_length的长度
        
        #这里的数据，是一批一批生成的
        yield batch_en_arr,batch_en_lengths,batch_de_arr,batch_de_lengths,batch_de_labels
        
        
        
        
def get_test_arr(text,seq_length):
    #变成"春 眠 不 觉 晓”
    text = ' '.join(text)
    words = text.split()
    arr = []
    #当前上联有多少个字
    len_word = len(words)
    #一般的写法是这个，将每一行的字符转为数字
    #arr = [word_to_int[word] for word in words]
    #但是在中文字表中，我们只取了一部分数据，如果有生僻字，就会出错
    #首先把每个字符在字典表里找到对应的序号，如果找不到，添加5000
    #这里把unknow解决了，unknow也指生僻字
    for word in words:
        #非生僻字
        if word in word_to_int:
            arr.append(word_to_int[word])
        #生僻字，记为5000
        else:
            arr.append(max_vocabs)
            
    #接下来就是padding
    #如果上联的长度，小于最大长度seq_length
    if(len_word<seq_length):
        #padding也是用5000代替
        #小于则填充，用5000代替
        #如[30,368,91,63,540,117,732,5000,5000,
        #5000,5000,5000,5000,5000,5000,5000,5000,
        #5000,5000,5000,5000,5000,5000,5000,5000,5000,5000,5000,5000,5000]
        arr+=[max_vocabs]*(seq_length-len_word)
    else:
        #超出部分，则截掉
        arr=arr[:seq_length]
        len_word = seq_length
        
    return np.array(arr)[None,:],np.array([len_word])


def id_to_word(id_):
    return int_to_word[id_]
    
    









