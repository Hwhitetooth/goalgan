import gan_network as nn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

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


def get_batch_disk(batch_size, r_min=0.3, r_max=0.6):
    R = np.random.uniform(r_min-0.1, r_max+0.1, size=batch_size)
    THETA = np.random.uniform(0, 2*np.pi, size=batch_size)
    X, Y = R*np.cos(THETA), R*np.sin(THETA)
    label_bool = (R > r_min) & (R < r_max)
    labels = np.zeros_like(label_bool, dtype=np.float32)
    labels[label_bool > 0] = 1
    goals = np.concatenate([X[:, None], Y[:, None]], axis=1)
    return goals, labels

#goals, labels = get_batch_disk(32)
#for i in range(32):
#    print(goals[i, 0] > 0, goals[i, 1] > 0, labels[i])


def scatter_plot(goals, j, i, r_min, r_max):
    f, ax = plt.subplots()
    for k in range(goals.shape[0]):
        x = goals[k,0]
        y = goals[k,1]
        if np.sqrt(x*x+y*y) < r_min:
            ax.scatter(x, y, color='red')
        elif np.sqrt(x*x+y*y) < r_max:
            ax.scatter(x, y, color='green')
        else:
            ax.scatter(x, y, color='blue')
    f.savefig('./samples/sample%s_%s.png' % (j,i))
    f.clf()


def assign_labels(goals, r_min, r_max):
    labels = []
    for k in range(goals.shape[0]):
        x = goals[k,0]
        y = goals[k,1]
        if (np.sqrt(x*x+y*y) < r_min) or (np.sqrt(x*x+y*y) >= r_max):
            labels.append(0)
        else:
            labels.append(1)
    return np.array(labels)
 

def main():
    i = 0
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    lsgan = nn.LSGAN(sess, 'lsgan')
    sess.run(tf.global_variables_initializer())

    # scatter_plot(lsgan.generate_goals(1000), 1300012810, 15126236, 0, 1)
    # exit()

    replay_buffer = deque(maxlen=1000)
    lsgan.train_init()
    for j in range(60):
        r_min = j * 0.05
        r_max = j * 0.05 + 1
        training_goals = []
        for replay_idx in range(min(500 // 3, len(replay_buffer))):
            training_goals.append(np.array(replay_buffer[np.random.randint(len(replay_buffer))]))
        training_goals.extend(lsgan.generate_goals(500 - len(training_goals)))

        filtered_goals = []
        for goal in training_goals:
            novel = True
            for exist_goal in filtered_goals:
                if np.sqrt(np.sum(np.square(goal - exist_goal))) < 0.1:
                    novel = False
                    break
            if novel:
                filtered_goals.append(goal)
        training_goals = np.array(filtered_goals)
        print("# of training goals:", training_goals.shape[0])

        scatter_plot(training_goals, j,-1, r_min, r_max)
        training_labels = assign_labels(training_goals, r_min, r_max)

        #reset GAN network
        sess.run(tf.global_variables_initializer())
        for i in range(2000) :
            n = training_goals.shape[0]
            indices = np.random.randint(0, n, n)
            d_loss, g_loss = lsgan.train_step(training_labels[indices], training_goals[indices])
            
            if i % 100 == 0:
                goals = lsgan.generate_goals(1000)
                scatter_plot(goals, j, i // 100, r_min, r_max)
            print(j, i, g_loss, d_loss)
        for zzb in range(len(training_goals)):
            gx = training_goals[zzb][0]
            gy = training_goals[zzb][1]
            if np.sqrt(gx*gx+gy*gy)<r_min:
                continue
            novel=True
            for old_gx, old_gy in replay_buffer:
                dx = old_gx - gx
                dy = old_gy - gy
                if np.sqrt(dx * dx + dy * dy) <= 0.1:
                    novel=False
                    break
            if novel:
                replay_buffer.append((gx, gy))

if __name__ == "__main__":
    main()
