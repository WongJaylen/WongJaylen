import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from self_Dense import Dense#导入自定义的dense层
from keras.utils.vis_utils import plot_model#画模型图
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'



def LSTM_textCNN(embedding_vector,m):

	#实例化一个张量
  main_input = tf.keras.layers.Input(shape=(100,), dtype='float64')

  #嵌入层
  embedder1 = tf.keras.layers.Embedding(m+1, 128, input_length=100,weights = [embedding_vector],trainable=False)
  embed1 = embedder1(main_input)
  
  #卷积层		
  conv_1 = tf.keras.layers.Conv1D(64, 3, padding='same', strides=1, activation='relu')(embed1)
  conv_2 = tf.keras.layers.Conv1D(64, 4, padding='same', strides=1, activation='relu')(embed1)
  conv_3 = tf.keras.layers.Conv1D(64, 5, padding='same', strides=1, activation='relu')(embed1)

  #池化层
  pool_1 = tf.keras.layers.MaxPooling1D(pool_size=38)(conv_1)
  pool_2= tf.keras.layers.MaxPooling1D(pool_size=37)(conv_2)
  pool_3 = tf.keras.layers.MaxPooling1D(pool_size=36)(conv_3)

  # 合并三个模型的输出向量
  cnn = concatenate([pool_1, pool_2, pool_3], axis=-1)

  #展平图层
  #flat = tf.keras.layers.Flatten()(cnn)
  #biLSTM层
  lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50))(cnn)

  #Dropout层，防止过拟合
  drop = tf.keras.layers.Dropout(0.4)(lstm)

  #全连接层
  main_output = tf.keras.layers.Dense(5, activation='softmax')(drop)

  #完成模型的建立
  model = tf.keras.Model(inputs=main_input, outputs=main_output)
  
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  
  return model 
  
  


#从文本预处理的模块中导入词的数量，词向量，用作训练、测试的序列
from text_preprocess import vocab_num,	embedding_vector,	pad_sequence_train,	pad_sequence_test,	one_hot_labels




#lstm模型
model=LSTM(embedding_vector,vocab_num)
#训练
model_train=model.fit(pad_sequence_train, one_hot_labels, batch_size=8001, epochs=50)   
 



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
df_predict.to_csv('data/submit.csv',index_label='id')
