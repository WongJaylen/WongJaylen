import tkinter as tk
import tensorflow as tf
import numpy as np
import pandas as pd
import re
from text_preprocess import stopwords,	remove_punctuation,	tokenizer#从自己的文本预处理文件中载入停词表，标点符号表和分词器

#载入用来分类文本的模型
model=tf.keras.models.load_model('data/LSTM_model.h5')


#将文本转换成数字序列
def text_process(text):
	
	#去掉每个句子的标点符号
	string=re.sub('[{}]'.format(remove_punctuation),"",text)
	
	#转换小写和分词
	words_seq=string.lower().split()
	
	words=[]#列表储存去词后的词语
	for word in words_seq:#把停用词去掉
		if word not in stopwords:
			words.append(word)#
	

	#对文本转换成数字序列		
	sequence=np.array(tokenizer.texts_to_sequences(words)).reshape((1,len(words)))
	
	#进行长度100的padding
	pad_sequence=tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100)
	
	return pad_sequence




	#创建主窗口
window = tk.Tk()
window.title('文本分类')
window.geometry('750x500')


label_title=tk.Label(window,	text='Text sentiment prediction',	font=('Arial', 20),	width=20,	height=2)
label_title.pack(pady=25)

label_tip=tk.Label(window,	text='请输入文本',	font=('Arial', 15),	width=15,	height=2)
label_tip.pack(padx=100)


e=tk.Entry(window,	width=50)#输入框
e.pack(padx=100,	pady=10)


#按钮执行的命令
def text_predict():
	
	text=e.get()#获取输入框的文本
	
	e.delete(0,"end")#清空输入框
	
	sequence = text_process(text)#对文本进行处理
	
	#用模型分类获得结果
	result = model.predict(sequence)
	result_labels = np.argmax(result, axis=1) 
	y_predict = list(map(str, result_labels))
	 
	w = tk.Tk()#另外建立一个窗口输出分类结果
	w.title('分类预测')
	w.geometry('300x150')
	
	label_1 = tk.Label(w,	text='该文本的分类结果为',	font=('Arial', 20),width=100,height=2)
	label_2 = tk.Label(w,	text=y_predict,	font=('Arial', 25),width=50,height=2)
	label_1.pack()
	label_2.pack(pady=15)
	
	w.mainloop()
	


#按钮
b = tk.Button(window, text='Enter',width=15, height=2,command=text_predict)
b.pack(pady=25)    


window.mainloop()



