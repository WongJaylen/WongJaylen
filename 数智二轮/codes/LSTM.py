import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from self_Dense import Dense#导入自定义的dense层
from keras.utils.vis_utils import plot_model#画模型图
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'



#lstm模型
def LSTM(embedding_vector,vocab_num):
	
	model = tf.keras.Sequential()
	
	#input层
	model.add(tf.keras.layers.Input(shape=(100,), dtype='float64'))
	
	#embedding层
	model.add(tf.keras.layers.Embedding(vocab_num+1, 128, input_length=100, weights = [embedding_vector], trainable=False))
	
	#LSTM层
	#model.add(tf.keras.layers.LSTM(50,dropout=0.2, recurrent_dropout=0.2))#LSTM
	model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50),merge_mode='concat'))#biLSTM
 
	#Dropout层
	model.add(tf.keras.layers.Dropout(0.6))
	
	#全连接层
	model.add(tf.keras.layers.Dense(5, activation='softmax'))
	#model.add(Dense(units=5,activation='softmax'))#调用自定义的全连接层

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  
	return model                       



#从文本预处理的模块中导入词的数量，词向量，用作训练、测试的序列
from text_preprocess import vocab_num,	embedding_vector,	pad_sequence_train,	pad_sequence_test,	one_hot_labels




#lstm模型
model=LSTM(embedding_vector,vocab_num)
#训练
model_train=model.fit(pad_sequence_train, one_hot_labels, batch_size=8001, epochs=50)   
 


#画出模型图
#plot_model(model,to_file='data/LSTMmodel.png',show_shapes=True,show_layer_names=False,rankdir='LR')

#模型的保存
#model.save('data/LSTM_model.h5')
#模型的读取
#model=tf.keras.models.load_model('data/LSTM_model.h5')

#绘图
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.plot(model_train.history['accuracy'], c='g', label='train')
plt.plot(model_train.history['val_accuracy'], c='b', label='validation')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Model accuracy')

plt.subplot(122)
plt.plot(model_train.history['loss'], c='g', label='train')
plt.plot(model_train.history['val_loss'], c='b', label='validation')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Model loss')

plt.show()



#测试集预测
result = model.predict(pad_sequence_test)
result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
y_predict = list(map(str, result_labels))

df_predict=pd.DataFrame(y_predict)
df_predict.columns=['label']
df_predict.to_csv('data/submit_LSTM.csv',index_label='id')
