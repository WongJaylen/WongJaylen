import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from self_Dense import Dense#导入自定义的dense层
from keras.utils.vis_utils import plot_model#画模型图
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'



#构建CNN模型
def textCNN(embedding_vector,vocab_num):
	
	#实例化一个张量
  main_input = tf.keras.layers.Input(shape=(100,), dtype='float64')

	#嵌入层
  embedder=tf.keras.layers.Embedding(vocab_num+1, 128, input_length=100,weights = [embedding_vector],trainable=False)
  embed = embedder(main_input)

	#卷积层		
		#滤波器数量（输出空间的维数）为64, 卷积窗口的长度分别为3,4,5， padding填充为'same'，让输入张量卷积后的得到的维度不变，输出(batch size,句子长度,64)			
  conv_1 = tf.keras.layers.Conv1D(64, 8, padding='same', strides=1, activation='relu')(embed)
  conv_2 = tf.keras.layers.Conv1D(64, 9, padding='same', strides=1, activation='relu')(embed)
  conv_3 = tf.keras.layers.Conv1D(64, 10, padding='same', strides=1, activation='relu')(embed)

	#池化层,池化窗口为38，步长默认为38
  pool_1 = tf.keras.layers.MaxPooling1D(pool_size=38)(conv_1)
  pool_2= tf.keras.layers.MaxPooling1D(pool_size=37)(conv_2)
  pool_3 = tf.keras.layers.MaxPooling1D(pool_size=36)(conv_3)
  
	#合并三个模型的输出向量
  cnn = tf.keras.layers.concatenate([pool_1, pool_2, pool_3], axis=-1)
  
	#展平图层
  flat = tf.keras.layers.Flatten()(cnn)

	#Dropout层，防止过拟合
  drop = tf.keras.layers.Dropout(0.6)(flat)

	#全连接层
  main_output = tf.keras.layers.Dense(5, activation='softmax')(drop)

	#完成模型的建立
  model = tf.keras.Model(inputs=main_input, outputs=main_output)

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  return model


#从文本预处理的模块中导入词的数量，词向量，用作训练、测试的序列
from text_preprocess import vocab_num,	embedding_vector,	pad_sequence_train,	pad_sequence_test,	one_hot_labels

#定义模型
model=textCNN(embedding_vector,vocab_num)

#模型训练
model_train=model.fit(pad_sequence_train, one_hot_labels, batch_size=8001, epochs=1)   


#画出模型图
#plot_model(model,to_file='data/textCNNmodel.png',show_shapes=True,show_layer_names=False,rankdir='LR')


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
df_predict.to_csv('data/submit_textCNN.csv',index_label='id')

