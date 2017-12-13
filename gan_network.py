import tensorflow as tf
import numpy as np

def get_vars(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

def get_batch_disk(batch_size, r_max=0, r_min=1):
    R = np.random.uniform(0, 1, size=batch_size)
    THETA = np.random.uniform(0, 2*np.pi, size=batch_size)
    X, Y = R*np.cos(THETA), R*np.sin(THETA)
    label_bool = (R < r_min) & (R > r_max)
    labels = np.zeros_like(label_bool, dtype=np.float32)
    labels[label_bool > 0] = 1
    goals = np.concatenate([X[:, None], Y[:, None]], axis=1)
    return goals, labels


class LSGAN(object):

    def __init__(self, sess, scope, g_learning_rate=0.0001, d_learning_rate=0.0001, beta1=0.5, reuse=None, is_training=True):
        self.scope = scope
        self.reuse = reuse
        self.sess = sess
        self.use_batchnorm = False

        self.inp_goal= tf.placeholder(tf.float32, [None, 2])
        self.inp_label = tf.placeholder(tf.float32, [None])
        self.inp_noise = tf.placeholder(tf.float32, [None, 4])

        self.GZ = self.make_generator_network(self.inp_noise)
        self.DX = tf.squeeze(self.make_discriminator_network(self.inp_goal),axis=[1])
        self.DGZ = tf.squeeze(self.make_discriminator_network(self.GZ, reuse=True),axis=[1])

        # self.a, self.b, self.c = -1, 1, 0
        self.a, self.b, self.c = 0, 1, 1
        self.a_ph, self.b_ph, self.c_ph = tf.placeholder(tf.float32), tf.placeholder(tf.float32), tf.placeholder(tf.float32)
        
        self.G_loss = tf.reduce_mean(tf.square(self.DGZ - self.c_ph))
        self.D_loss = tf.reduce_mean(self.inp_label * tf.square(self.DX - self.b_ph) + (1 - self.inp_label) * tf.square(self.DX - self.a_ph)) + tf.reduce_mean(tf.square(self.DGZ - self.a_ph))

        self.g_loss = 1
        self.d_loss = 1
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.G_train = tf.train.RMSPropOptimizer(learning_rate=g_learning_rate).minimize(self.G_loss, var_list=get_vars('G'))
            self.D_train = tf.train.RMSPropOptimizer(learning_rate=d_learning_rate).minimize(self.D_loss, var_list=get_vars('D'))


    def bn_func_gen(self, x):
        if self.use_batchnorm:
            return tf.nn.leaky_relu(tf.layers.batch_normalization(x))
        else:
            return tf.nn.leaky_relu(x, alpha=0.0)

    def bn_func_disc(self, x):
        if self.use_batchnorm:
            return tf.nn.leaky_relu(tf.layers.batch_normalization(x))
        else:
            return tf.nn.leaky_relu(x, alpha=0.0)
     
    def train_init(self):
        olda, oldb, oldc = self.a, self.b, self.c
        self.a, self.b, self.c = 0, 1, 1
        training_goals, training_labels = get_batch_disk(1000)
        for i in range(800):
            indices = np.random.randint(0, 1000, 100)
            d_loss, g_loss = self.train_step(training_labels[indices], training_goals[indices])
        self.a, self.b, self.c = olda, oldb, oldc


    def train_step(self, labels, goals):
        batch_size = len(goals) 
        noise = np.random.normal(0, 1, size=[batch_size, 4])
        [_, self.d_loss, dx, dgz] = self.sess.run([self.D_train, self.D_loss, self.DX, self.DGZ], feed_dict={self.inp_goal: goals, self.inp_label: labels, self.inp_noise: noise, self.a_ph: self.a, self.b_ph: self.b})
        """
        for _ in range(int((self.g_loss - self.d_loss)/0.2)):
            print('train_discriminator',int((self.g_loss - self.d_loss)/0.2)) 
            noise = np.random.normal(0, 1, size=[batch_size, 4])
            [_, self.d_loss] = self.sess.run([self.D_train, self.D_loss], feed_dict={self.inp_goal: goals, self.inp_label: labels, self.inp_noise: noise})
        for _ in range(int((self.d_loss - self.g_loss)/0.2)) :
            print('train_generator', int((self.d_loss - self.g_loss)/0.2))
            noise = np.random.normal(0, 1, size=[batch_size, 4])
            [_, self.g_loss] = self.sess.run([self.G_train, self.G_loss], feed_dict={self.inp_noise: noise})
        """
        noise = np.random.normal(0, 1, size=[batch_size, 4])
        [_, self.g_loss] = self.sess.run([self.G_train, self.G_loss], feed_dict={self.inp_noise: noise, self.c_ph: self.c})
        return self.g_loss, self.d_loss

    def generate_goals(self, num_goals):
        noise = np.random.normal(0, 1, size=[num_goals, 4])
        [gen] = self.sess.run([self.GZ], feed_dict={self.inp_noise: noise})
        return gen
    

    def make_generator_network(self, noise, reuse=False):
        with tf.variable_scope('G', reuse=reuse):
            x = self.bn_func_gen(tf.layers.dense(noise, 128, kernel_initializer=tf.initializers.random_uniform(minval=-np.sqrt(30/132),maxval=np.sqrt(30/132))))
            x = self.bn_func_gen(tf.layers.dense(x, 128, kernel_initializer=tf.initializers.random_uniform(minval=-np.sqrt(30/256),maxval=np.sqrt(30/256))))
            x = tf.layers.dense(x, 2, kernel_initializer=tf.initializers.random_uniform(minval=-np.sqrt(30/130),maxval=np.sqrt(30/130)))
        return x

    def make_discriminator_network(self, goal, reuse=False):
        with tf.variable_scope('D', reuse=reuse):
            x = self.bn_func_disc(tf.layers.dense(goal, 256, kernel_initializer=tf.initializers.random_uniform(minval=-np.sqrt(30/258),maxval=np.sqrt(30/258))))
            x = self.bn_func_disc(tf.layers.dense(x, 256, kernel_initializer=tf.initializers.random_uniform(minval=-np.sqrt(30/512),maxval=np.sqrt(30/512))))
            x = tf.layers.dense(x, 1, kernel_initializer=tf.initializers.random_uniform(minval=-np.sqrt(30/257),maxval=np.sqrt(30/257)))
        return x


