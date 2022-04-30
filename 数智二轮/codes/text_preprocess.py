import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
from gensim.models import word2vec
import tensorflow as tf


'''#将训练集和测试集文本合并一起
file=pd.read_csv('data/train.csv')
df=pd.DataFrame(file)

df['text'].to_csv('data/sum.csv')

file=pd.read_csv('data/test.csv')
df=pd.DataFrame(file)
df['text'].to_csv('data/sum.csv',mode='a',header=False)'''


#读取文件，对训练集和测试集的全部数据同时处理
file=pd.read_csv('data/sum.csv')
df=pd.DataFrame(file)


#要去除的标点符号
remove_punctuation = string.punctuation


#根据TFIDF,自行创建的停词表
stopwords=['a','am','about','above','an','and','are','about','against','anywhere','after','as','at','across','area',
'be', 'between','because', 'been', 'before', 'being','by',
'for','from',
'had','he', 'him', 'his', 'himself','her', 'hers','herself','how','here','has','have',
'is','i','im','in','it', 'its', 'itself','if','into','ll','me', 'my', 'myself','night',
'what', 'which', 'who', 'whom','with','we','was','when', 'while','were','why',
'this', 'that', 'these', 'those', 'they', 'them', 'their', 'theirs', 'themselves','then','the','to',
'of','on','our', 'ours', 'ourselves','out','off','or',
'so','she','since','said','up','us','ve',
'you', 'your', 'yours','yourself', 'yourselves',]



texts=[]#储存所有预处理后的文档

#遍历所有文档样本进行预处理
for i in range (len(df)):

	#去掉每个句子的标点符号
	string=re.sub('[{}]'.format(remove_punctuation),"",df['text'][i])
	
	#转换小写和分词
	words_seq=string.lower().split()
	
	words=[]#列表储存去词后的词语
	
	#把停用词去掉
	for word in words_seq:
		if word not in stopwords:
			words.append(word)#储存的是单个文档的词
			
	texts.append(words)#储存的是全部文档

#使用keras库的分词器进行向量化文本
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',oov_token = '<OOV>')
tokenizer.fit_on_texts(texts)

#得到每个词的编号,编号从1开始
vocab=tokenizer.word_index 

vocab_num=len(vocab)#词的数量

#对文本转换成数字序列
sequence_train=tokenizer.texts_to_sequences(texts[:8001])
sequence_test=tokenizer.texts_to_sequences(texts[8001:])
#进行相同长度的padding
pad_sequence_train=tf.keras.preprocessing.sequence.pad_sequences(sequence_train, maxlen=100)
pad_sequence_test=tf.keras.preprocessing.sequence.pad_sequences(sequence_test, maxlen=100)



#加载训练得到的词向量
word_vector=word2vec.Word2Vec.load("data/word2vec_model_sum.model")

#嵌入层的权重矩阵，预训练的词向量中没有出现的词用0向量表示  
embedding_vector=np.zeros((vocab_num+1,128))
#对每个词进行词向量表示
for word, i in vocab.items():
    try:        
        embedding_vector[i] = word_vector.wv[str(word)]
        
    except KeyError:
        continue



#将训练集的标签值化为one hot序列
file=pd.read_csv('data/train.csv')
df_train=pd.DataFrame(file)
one_hot_labels = tf.keras.utils.to_categorical(df_train['label'], num_classes=5)




'''#将训练样本以不同方式截断和填充来增加样本数量
pad_sequence_train_1=tf.keras.preprocessing.sequence.pad_sequences(sequence_train, maxlen=100)
pad_sequence_train_2=tf.keras.preprocessing.sequence.pad_sequences(sequence_train, maxlen=100,padding='post',truncating='post')
pad_sequence_train=np.insert(pad_sequence_train_1 ,0, values=pad_sequence_train_2 , axis=0)

#增加样本标签
df=pd.DataFrame(np.insert(df_train['label'].values , 0, values=df_train['label'].values, axis=0))

one_hot_labels = tf.keras.utils.to_categorical(df[0], num_classes=5)'''
