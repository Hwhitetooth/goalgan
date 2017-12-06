import gan_network as nn
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_batch_disk2(batch_size, r_max=0.2, r_min=0.5):
    #B = np.random.binomial(1, 0.5, size=batch_size)
    R = np.random.uniform(0.2, 0.5, size=batch_size)
    THETA = np.random.uniform(0, 2*np.pi, size=batch_size)
    X, Y = R*np.cos(THETA), R*np.sin(THETA)
    label_bool = (R < r_min) & (R > r_max)
    labels = np.ones_like(label_bool)
    #labels[label_bool > 0] = 1
    goals = np.concatenate([X[:, None], Y[:, None]], axis=1)
    return goals, labels


def get_batch_disk(batch_size, r_max=0.2, r_min=0.5):
    R = np.random.uniform(0.2, 0.5, size=batch_size)
    THETA = np.random.uniform(0, 2*np.pi, size=batch_size)
    X, Y = R*np.cos(THETA), R*np.sin(THETA)
    label_bool = (R < r_min) & (R > r_max)
    labels = np.zeros_like(label_bool, dtype=np.float32)
    labels[label_bool > 0] = 1
    goals = np.concatenate([X[:, None], Y[:, None]], axis=1)
    return goals, labels

#goals, labels = get_batch_disk(32)
#for i in range(32):
#    print(goals[i, 0] > 0, goals[i, 1] > 0, labels[i])


def scatter_plot(goals, i):
    f, ax = plt.subplots()
    ax.scatter(goals[:, 0], goals[:, 1], color='blue')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    f.savefig('./samples/sample%s.png' % i)



def main():
    i = 0
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    lsgan = nn.LSGAN(sess, 'lsgan')
    sess.run(tf.global_variables_initializer())
    while True:

        goals, labels = get_batch_disk(64)
        d_loss = lsgan.train_discriminator(labels, goals)

        for _ in range(2):
            goals, labels = get_batch_disk(64)
            g_loss = lsgan.train_generator(labels, goals)

        if i % 500 == 0:
            goals = lsgan.generate_goals(1000)
            scatter_plot(goals, i // 500)
        #img = cv2.resize(255*generated[0][::-1], (400, 400), interpolation=cv2.INTER_NEAREST)
        i += 1
        print(i, g_loss, d_loss)
        #if i % 10 == 0:
        #    cv2.imwrite('./sample.png', img)


main()