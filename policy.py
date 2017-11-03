import numpy as np
import tensorflow as tf
import distributions

class MLPPolicy(object):
    def __init__(self, sess, input_shape, action_space, scope = None):
        output_size = action_space.shape[0]
        self.sess = sess
        scope = tf.get_variable_scope() if not scope else scope
        batch_shape = [None] + list(input_shape)
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(tf.float32, batch_shape, "inputs")
            with tf.contrib.framework.arg_scope([tf.contrib.layers.fully_connected],
                    activation_fn = tf.nn.tanh, weights_initializer = normc_initializer(1.0)):
                fc1 = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(self.inputs), 64, scope = "v_fc1")
                fc2 = tf.contrib.layers.fully_connected(fc1, 64, scope = "v_fc2")
                self.v = tf.squeeze(tf.contrib.layers.fully_connected(fc2, 1, activation_fn = None, scope = "v"), [1])
            with tf.contrib.framework.arg_scope([tf.contrib.layers.fully_connected],
                    activation_fn = tf.nn.tanh, weights_initializer = normc_initializer(1.0)):
                fc1 = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(self.inputs), 64, scope = "pi_fc1")
                fc2 = tf.contrib.layers.fully_connected(fc1, 64, scope = "pi_fc2")
                self.miu= tf.contrib.layers.fully_connected(fc2, output_size, activation_fn = None, weights_initializer = normc_initializer(0.01), scope = "pi")
            self.logsigma = tf.Variable(tf.zeros([output_size]), name = "logsigma")
            self.pi = distributions.Normal(self.miu, self.logsigma)
            self.a = self.pi.sample()
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

    def get_v(self, inputs):
        v = self.sess.run(self.v, feed_dict = {self.inputs: inputs})
        return v[0]
    
    def get_a_and_v(self, inputs):
        a, v = self.sess.run([self.a, self.v], feed_dict = {self.inputs: inputs})
        return a[0], v[0]

class CNNPolicy(object):
    def __init__(self, sess, input_shape, action_space, scope):
        output_size = action_space.n
        self.sess = sess
        scope = tf.get_variable_scope() if not scope else scope
        batch_shape = [None] + list(input_shape)
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(tf.float32, batch_shape, "inputs")
            x = self.inputs / 255.0
            conv1 = tf.contrib.layers.conv2d(x, 16, 8, 4, "VALID", scope = "conv1")
            conv2 = tf.contrib.layers.conv2d(conv1, 32, 4, 2, "VALID", scope = "conv2")
            fc1 = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(conv2), 256, scope = "fc1")
            self.logits = tf.contrib.layers.fully_connected(fc1, output_size, activation_fn = None, scope = "logits")
            self.v = tf.squeeze(tf.contrib.layers.fully_connected(fc1, 1, activation_fn = None, scope = "v"), [1])
            self.pi = distributions.Categorical(self.logits)
            self.a = self.pi.sample()
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

    def get_v(self, inputs):
        v = self.sess.run(self.v, feed_dict = {self.inputs: inputs})
        return v[0]

    def get_a_and_v(self, inputs):
        a, v = self.sess.run([self.a, self.v], feed_dict = {self.inputs: inputs})
        return a[0], v[0]

def normc_initializer(std = 1.0):
    def _initializer(shape, dtype = None, partition_info = None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis = 0, keepdims = True))
        return tf.constant(out)
    return _initializer
