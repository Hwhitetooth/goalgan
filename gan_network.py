import tensorflow as tf
import numpy as np

def leaky_relu(x, alpha=0.0001):
    return tf.maximum(x, alpha*x)
def get_vars(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

class LSGAN(object):

    def __init__(self, sess, scope, g_learning_rate=0.0005, d_learning_rate=0.0005, beta1=0.5, reuse=None, is_training=True):
        self.scope = scope
        self.reuse = reuse
        self.is_training = is_training
        self.sess = sess
        self.use_batchnorm = True

        #self.inp_noise = tf.placeholder(tf.float32, [None, 1024])
        self.inp_goal= tf.placeholder(tf.float32, [None, 2])
        self.inp_label = tf.placeholder(tf.float32, [None])
        self.inp_noise = tf.placeholder(tf.float32, [None, 100])

        self.GZ = self.make_generator_network(self.inp_noise, reuse=False)
        self.DX = self.make_discriminator_network(self.inp_goal, reuse=False)
        self.DGZ = self.make_discriminator_network(self.GZ, reuse=True, is_training=True)

        self.a, self.b, self.c = -1, 1, 0

        #self.G_loss = 0.5 * tf.reduce_mean(tf.square(self.DGZ - self.c))
        #self.D_loss = 0.5 * tf.reduce_mean(self.inp_label * tf.square(self.DX - self.b)+ (1 - self.inp_label) * tf.square(self.DX - self.a) + tf.square(self.DGZ - self.a))
        self.G_loss = tf.reduce_mean(tf.square(self.DGZ - self.c))
        self.D_loss = tf.reduce_mean(self.inp_label * tf.square(self.DX - self.b) + (1 - self.inp_label) * tf.square(self.DX - self.a) + tf.square(self.DGZ - self.a))

        
        self.init_G_loss = tf.reduce_mean(tf.square(self.DGZ-1))
        self.init_D_loss = tf.reduce_mean(tf.square(self.DX-1)+tf.square(self.DGZ))        

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.G_train = tf.train.RMSPropOptimizer(learning_rate=g_learning_rate).minimize(self.G_loss, var_list=get_vars('G'))
            self.D_train = tf.train.RMSPropOptimizer(learning_rate=d_learning_rate).minimize(self.D_loss, var_list=get_vars('D'))

        init_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(init_update_ops):
            self.init_G_train = tf.train.AdamOptimizer(learning_rate=g_learning_rate, beta1=beta1).minimize(self.init_G_loss, var_list=get_vars('G'))            
            self.init_D_train = tf.train.AdamOptimizer(learning_rate=d_learning_rate, beta1=beta1).minimize(self.init_D_loss, var_list=get_vars('D'))



    def bn_func_gen(self, x, training=True):
        if self.use_batchnorm and False:
            return leaky_relu(tf.layers.batch_normalization(x, training=training))
        else:
            return leaky_relu(x)
        #return tf.nn.relu(x)

    def bn_func_disc(self, x, training=True):
        if self.use_batchnorm:
            #return leaky_relu(tf.layers.batch_normalization(x, training=training))
            return leaky_relu(tf.layers.batch_normalization(x, training=training))
        else:
            #return leaky_relu(x)
            return leaky_relu(x)
     
    def train_init(self):
        goals = []
        for i in range(20):
            gx = (i+1) * 0.03 + 0.1
            for j in range(20): 
                gy = (j+1) * 0.03 + 0.1
                if np.sqrt(gx*gx + gy*gy)<0.55 and len(goals)<100:
                    print('Goal (', gx, ',', gy, ') Distance', np.sqrt(gx*gx + gy*gy))
                    goals.append([gx, gy])
        for i in range(10000):
            noise = np.random.normal(0, 1, size=[100, 4])
            [_, d_loss] = self.sess.run([self.init_D_train, self.init_D_loss], feed_dict={self.inp_goal: goals, self.inp_noise: noise})
            [_, g_loss] = self.sess.run([self.init_G_train, self.init_G_loss], feed_dict={self.inp_goal: goals, self.inp_noise: noise})
            #print('G loss', g_loss, 'D loss', d_loss)



    """

    def train_init(self):
        goals = []
        labels = []
        for i in range(10):
            gx = (i+1) * 0.05
            for j in range(10): 
                gy = (j+1) * 0.05
                goals.append([gx, gy])
        for i in range(100):
            d = np.sqrt(goals[i][0]*goals[i][0] + goals[i][1]*goals[i][1])
            if d < 0.5:
                labels.append(1)
            else:
                labels.append(0)
            print('Goal (', goals[i][0], ',', goals[i][1], ') Distance', d, 'Label', labels[i])
        for i in range(10000):
            noise = np.random.normal(0, 1, size=[100, 4])
            [_, d_loss] = self.sess.run([self.D_train, self.D_loss], feed_dict={self.inp_goal: goals, self.inp_label: labels, self.inp_noise: noise})
            [_, g_loss] = self.sess.run([self.G_train, self.G_loss], feed_dict={self.inp_goal: goals, self.inp_label: labels, self.inp_noise: noise})
            print('G loss', g_loss, 'D loss', d_loss)
    """

    def train_discriminator(self, labels, goals):
        batch_size = len(goals)
        noise = np.random.normal(0, 1, size=[batch_size, 100])
        [_, d_loss] = self.sess.run([self.D_train, self.D_loss], feed_dict={self.inp_goal: goals, self.inp_label: labels, self.inp_noise: noise})
        #print('G loss', g_loss, 'D loss', d_loss)
        return d_loss


    def train_generator(self, labels, goals):
        batch_size = len(goals)
        noise = np.random.normal(0, 1, size=[batch_size, 100])
        [_, g_loss] = self.sess.run([self.G_train, self.G_loss], feed_dict={self.inp_noise: noise, self.inp_goal: goals, self.inp_label: labels})
        return g_loss
    def generate_goals(self, num_goals):
        noise = np.random.normal(0, 1, size=[num_goals, 100])
        [gen] = self.sess.run([self.GZ], feed_dict={self.inp_noise: noise})
        return gen
    

    def make_generator_network(self, noise, reuse):
        with tf.variable_scope('G', reuse=reuse):
            x = self.bn_func_gen(tf.layers.dense(noise, 500), training=self.is_training)
            x = self.bn_func_gen(tf.layers.dense(x, 500), training=self.is_training)
            x = self.bn_func_gen(leaky_relu(tf.layers.dense(x, 500)))
            # x = self.bn_func_gen(tf.layers.dense(x, 100, activation=None), training=self.is_training)
            x = tf.layers.dense(x, 2)
            #x = tf.clip_by_value(x, 0,5)
            #x = self.bn_func_gen(tf.layers.conv2d_transpose(tf.reshape(x, [-1, 8, 8, 256]), 256, 5, 2, padding='SAME', activation=None), training=self.is_training)
            #x = self.bn_func_gen(tf.layers.conv2d_transpose(x, 256, 5, 1, padding='SAME', activation=None), training=self.is_training) # 16 x 16
            #x = self.bn_func_gen(tf.layers.conv2d_transpose(x, 256, 5, 2, padding='SAME', activation=None), training=self.is_training)
            #x = tf.nn.sigmoid(tf.layers.conv2d_transpose(x, 3, 5, 1, padding='SAME', activation=None))  # 32 x 32

            #x = self.bn_func(tf.layers.conv2d_transpose(x, 256, 3, 1, activation=tf.nn.relu), training=self.is_training)
            #x = tf.layers.conv2d_transpose(x, 3, 5, 2, padding='SAME', activation=tf.nn.sigmoid)
            #print(x)
        return x

    def make_discriminator_network(self, goal, reuse, is_training=True):
        with tf.variable_scope('D', reuse=reuse):
            x = self.bn_func_disc(tf.layers.dense(goal, 256), training=self.is_training and is_training)
            x = self.bn_func_disc(tf.layers.dense(x, 256), training=self.is_training and is_training)
            x = self.bn_func_disc(tf.layers.dense(x, 256), training=self.is_training and is_training)
            x = tf.layers.dense(x, 1)
            x = tf.nn.tanh(x)
            #x = tf.layers.conv2d(image, 64, 5, 2, padding='SAME', activation=leaky_relu) # 16
            #x = self.bn_func_disc(tf.layers.conv2d(x, 64, 5, 2, padding='SAME', activation=None), training=self.is_training) # 8
            #x = self.bn_func_disc(tf.layers.conv2d(x, 64, 3, 2, padding='SAME', activation=None), training=self.is_training)
            #x = tf.layers.dense(tf.reshape(x, [-1, 4*4*64]), 1, activation=tf.nn.tanh)
        return x


