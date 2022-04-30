import re
import string
from gensim.models import word2vec
import pandas as pd


'''#计算文档的tf-idf值
from sklearn.feature_extraction.text import TfidfVectorizer
file=pd.read_csv('data/sum.csv')
df=pd.DataFrame(file)

vectorizer=TfidfVectorizer(max_features=200)
X = vectorizer.fit_transform(df['text'])
print(vectorizer.get_feature_names())'''



																	#训练词向量  
                            
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

#要去除的标点符号
remove_punctuation = string.punctuation



#file=pd.read_csv('data/train.csv')#只对训练集进行word2vec处理
file=pd.read_csv('data/sum.csv')#对所有样本进行word2vec处理
df=pd.DataFrame(file)


texts=[]#储存所有预处理后的文档

#遍历所有文档样本进行预处理
for i in range(len(df)):
	#string=re.sub(remove_punctuation, "",df['text'][i])#将文档去掉标点符号
	string=re.sub('[{}]'.format(remove_punctuation),"",df['text'][i])
	
	words_seq=string.lower().split()#转换小写和分词
	
	words=[]#列表储存分词后的词语

	
	for word in words_seq:#把停用词去掉
		if word not in stopwords:
			words.append(word)
			
	texts.append(words)
 




#训练word2vec模型并保存
model = word2vec.Word2Vec(texts, sg=1, vector_size=128, window=5, min_count=5, negative=3, sample=0.001, hs=1)

model.save("data/word2vec_model_sum.model")



#加载和输出词向量
model=word2vec.Word2Vec.load("word2vec_model.model")
#print(model.wv['food'])
