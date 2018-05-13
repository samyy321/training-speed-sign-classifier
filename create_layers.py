import tensorflow as tf

def create_conv(input_layer, filters_nb, kernel_size, dropout_pct):
	"""
	Create convolution layer with RELU activation function.
	"""
	layer_shape = int(input_layer.get_shape()[-1])
	conv_W = tf.Variable(tf.truncated_normal(shape=(kernel_size, kernel_size, layer_shape, filters_nb), stddev=0.1))
	conv_b = tf.Variable(tf.zeros(filters_nb))

	conv = tf.nn.conv2d(input_layer, conv_W, strides=[1, 1, 1, 1], padding='VALID') + conv_b
	conv = tf.nn.relu(conv)
	conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	return tf.nn.dropout(conv, dropout_pct)

def create_fc_layer(conv, neurons_count, dropout_pct):
	"""
	Create fully connected layer.
	"""
	layer_shape = (int(conv.get_shape()[1]), neurons_count)
	weights = tf.Variable(tf.truncated_normal(shape=layer_shape, stddev=0.1))
	biases = tf.Variable(tf.zeros(neurons_count))
	fc_layer = tf.matmul(conv, weights) + biases
	return tf.nn.dropout(tf.nn.relu(fc_layer), dropout_pct)
