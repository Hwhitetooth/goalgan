import tensorflow as tf
import numpy as np

class Normal(object):
    def __init__(self, mean, logstd):
        self.distribution = tf.distributions.Normal(mean, tf.exp(logstd))

    def sample(self):
        return self.distribution.sample()

    def log_prob(self, x):
        return tf.reduce_sum(self.distribution.log_prob(x), axis = -1)

    def entropy(self):
        return tf.reduce_sum(self.distribution.entropy(), axis = -1)

class Categorical(object):
    def __init__(self, logits):
        self.distribution = tf.distributions.Categorical(logits = logits)

    def sample(self):
        return self.distribution.sample()

    def log_prob(self, x):
        return self.distribution.log_prob(x)

    def entropy(self):
        return self.distribution.entropy()
