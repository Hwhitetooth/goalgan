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


def get_batch_disk(batch_size, r_min=0.3, r_max=3):
    R = np.random.uniform(r_min, r_max, size=batch_size)
    THETA = np.random.uniform(0, 2*np.pi, size=batch_size)
    X, Y = R*np.cos(THETA), R*np.sin(THETA)
    label_bool = (R > r_min) & (R < r_max)
    labels = np.zeros_like(label_bool, dtype=np.float32)
    labels[label_bool > 0] =1 
    goals = np.concatenate([X[:, None], Y[:, None]], axis=1)
    return goals, labels

#goals, labels = get_batch_disk(32)
#for i in range(32):
#    print(goals[i, 0] > 0, goals[i, 1] > 0, labels[i])


def scatter_plot(goals, j, i, r_min, r_max):
    f, ax = plt.subplots()
    R = np.sqrt(goals[:,0]**2 + goals[:,1]**2)
    red_goals = goals[R < r_min]
    green_goals = goals[(R > r_min) & (R < r_max)]
    blue_goals = goals[R > r_max]
    ax.scatter(red_goals[:, 0], red_goals[:, 1], color='red')
    ax.scatter(green_goals[:, 0], green_goals[:, 1], color='green')
    ax.scatter(blue_goals[:, 0], blue_goals[:, 1], color='blue')
    '''for k in range(goals.shape[0]):
        x = goals[k,0]
        y = goals[k,1]
        if np.sqrt(x*x+y*y) < r_min:
            ax.scatter(x, y, color='red')
        elif np.sqrt(x*x+y*y) < r_max:
            ax.scatter(x, y, color='green')
        else:
            ax.scatter(x, y, color='blue')'''
    f.savefig('./samples/sample%s_%s.png' % (j,i))



def main():
    i = 0
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    lsgan = nn.LSGAN(sess, 'lsgan')
    sess.run(tf.global_variables_initializer())
    r_min = 0
    r_max=1
    training_goals, training_labels = get_batch_disk(1000, r_min, r_max)
    scatter_plot(training_goals, 0,-1, r_min, r_max)
    for i in range(0,10000) :
        indices = np.random.randint(0, 1000, 1000)
        d_loss, g_loss = lsgan.train_step(training_labels[indices], training_goals[indices])

        if i % 200 == 0:
            goals = lsgan.generate_goals(1000)
            scatter_plot(goals, 0, i // 200, r_min, r_max)
        print(0, i, 'generator loss', g_loss, 'discriminator loss', d_loss)


main()
