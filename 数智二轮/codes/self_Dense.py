import tensorflow as tf
from keras import backend as K

                                                                 #自定义Dense层
class Dense(tf.keras.layers.Layer):
	
	def __init__(self,	units,	activation,		kernel_initializer='glorot_uniform',	bias_initializer='zeros',	**kwargs):	

		super(Dense, self).__init__(**kwargs)#继承	
		self.units = units#输出单元
		self.activation = tf.keras.activations.get(activation)#激活函数
		self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)#权重初始值
		self.bias_initializer = tf.keras.initializers.get(bias_initializer)#偏置初始值

	
	
	def build(self, input_shape):
		#权重
		self.w = self.add_weight(	shape=(input_shape[-1],self.units),		initializer=self.kernel_initializer,	name='weight',)#add_weight()创建权重的快捷方式
		#偏置
		self.bias = self.add_weight(shape=(self.units,),	initializer=self.bias_initializer,	 name='bias',)

		#self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: input_shape[-1]})
		self.built = True


	def call(self, inputs):
		output = K.dot(inputs, self.w)
		output = K.bias_add(output, self.bias, data_format='channels_last')
		
		if self.activation is not None:
			output = self.activation(output)
    
		return output


	def compute_output_shape(self, input_shape):#输入(batchsize，dim),输出(batchsize,units)
		assert input_shape and len(input_shape) >= 2

		assert input_shape[-1]

		output_shape = list(input_shape)
	
		output_shape[-1] = self.units

		return tuple(output_shape)
