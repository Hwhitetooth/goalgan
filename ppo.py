import numpy as np
import tensorflow as tf
import gym, roboschool
from utils import Dataset
import logger
from collections import deque
from env_wrapper import Env
import gan_network as nn


def rollout(env, pi, goals, episodes, render = False):
    s_batch, a_batch, r_batch, v_batch, done_batch = [], [], [], [], []
    score = []
    for gx, gy in goals:
        s = env.reset(gx, gy)
        step = 0
        ep = 0
        total_reward = 0
        while True:
            a, v = pi.get_a_and_v(s[None])
            if np.any(np.isnan(a)):
                print("WARNING: found NaN in actions", a)
                a = env.action_space.sample()
            s_next, r, done, _ = env.step(np.array(a))
            if render:
                env.render()
            s_batch.append(s)
            a_batch.append(a)
            r_batch.append(r)
            v_batch.append(v)
            done_batch.append(done)
            total_reward += r
            s = s_next
            step += 1
            if done:
                ep += 1
                s = env.reset()
                if episodes is not None and ep == episodes:
                    break
        g_score = total_reward
        score.append(g_score)
    return {"s": np.array(s_batch),
            "a": np.array(a_batch),
            "r": np.array(r_batch),
            "v": np.array(v_batch),
            "done": np.array(done_batch)}, score


def process_data(raw, gamma, lamda):
    s_batch, a_batch, r_batch, v_batch, done_batch = raw["s"], raw["a"], raw["r"], raw["v"], raw["done"]
    horizon = s_batch.shape[0]
    adv_batch = np.zeros(horizon, "float32")
    v_next = adv_next = 0
    for i in reversed(range(horizon)):
        done = done_batch[i]
        r = r_batch[i]
        v = v_batch[i]
        v_next = 0 if done else v_next
        delta = r + gamma * v_next - v
        adv_next = 0 if done else adv_next
        adv_batch[i] = delta + gamma * lamda * adv_next
        v_next = v_batch[i]
        adv_next = adv_batch[i]
    g_batch = v_batch + adv_batch
    adv_batch = (adv_batch - adv_batch.mean()) / adv_batch.std()
    data = raw
    data.pop("r")
    data.pop("v")
    data.pop("done")
    data["g"] = g_batch
    data["adv"] = adv_batch
    return data


def train(env_name,
        sess,
        policy_fn,
        logdir,
        gamma = 0.998,
        lamda = 0.995,
        batch_size = 64,
        epochs = 1,
        lr = 3E-4,
        clip_eps = 0.2,
        max_iters = 1000,
        eps = 0.5,
        num_goals = 100,
        r_min = 0.2,
        r_max = 0.8,
        render = False):
    env = gym.make(env_name)
    pi = policy_fn(sess, env.observation_space.shape, env.action_space, "pi")
    pi_old = policy_fn(sess, env.observation_space.shape, env.action_space, "pi_old")
    update_old = [tf.assign(var_old, var_new) for var_old, var_new in zip(pi_old.vars, pi.vars)]

    # Loss of surrogate policy objective.
    a_taken = tf.placeholder(pi.a.dtype, pi.a.shape)
    ratio = tf.exp(pi.pi.log_prob(a_taken) - pi_old.pi.log_prob(a_taken))
    adv_sample = tf.placeholder(tf.float32, [None])
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_sample, tf.clip_by_value(ratio, 1 - clip_eps, 1 + clip_eps) * adv_sample))

    # Loss of value function.
    v_sample = tf.placeholder(tf.float32, [None])
    v_loss = tf.reduce_mean(tf.square(v_sample - pi.v))

    total_loss = pi_loss + v_loss

    # Optimization.
    step = tf.placeholder(tf.int32)
    opt = tf.train.AdamOptimizer(lr, epsilon=1E-5)
    train_op = opt.minimize(total_loss, var_list=pi.vars)

    # LSGAN.
    lsgan = nn.LSGAN(sess, "lsgan")

    # Training.
    sess.run(tf.global_variables_initializer())
    lsgan.train_init()
    
    def update_policy(env, goals, iterations=5):
        scores = []
        for it in range(iterations):
            raw_data, score = rollout(env, pi, goals, episodes=1, render=render)
            scores.append(score)
            data = Dataset(process_data(raw_data, gamma, lamda), batch_size)
            sess.run(update_old)
            for _ in range(epochs):
                for batch in data:
                    train_feed = {pi.inputs: batch["s"], pi_old.inputs: batch["s"], a_taken: batch["a"], \
                        adv_sample: batch["adv"], v_sample: batch["g"], step: 0}
                    sess.run(train_op, train_feed)
        scores = np.array(scores)
        scores = np.mean(scores, axis=0)
        scores = list(scores)
        return zip(scores, goals)

    env = Env(env_name, eps=0.5)
    
    replay_buffer = deque(maxlen = num_goals)
    for it in range(max_iters):
        cand_goals = []
        for replay_idx in range(min(num_goals // 3, len(replay_buffer))):
            cand_goals.append(replay_buffer[np.random.randint(len(replay_buffer))])
        cand_goals.extend(lsgan.generate_goals(num_goals-len(cand_goals)))

        cand_goals = np.array(cand_goals)
        filtered_goals = []
        for goal in cand_goals:
            novel = True
            for exist_goal in filtered_goals:
                if np.sqrt(np.sum(np.square(goal - exist_goal))) < eps:
                    novel = False
                    break
            if novel:
                filtered_goals.append(goal)
        goals = np.array(filtered_goals)

        abs_goals = [(abs(goal[0]), abs(goal[1])) for goal in goals]
        results = list(update_policy(env, abs_goals, 5)) # This is slow!!!

        labels = [int(score >= r_min and score <= r_max) for score, (gx, gy) in results]
        sess.run(tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G')))
        sess.run(tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='D')))
        for _ in range(2000):
            lsgan.train_step(np.array(labels), goals)

        d_min, d_max = 1E10, 0
        scores = []
        for score, (gx, gy) in results:
            scores.append(score)
            if score >= r_min and score < r_max:
                d_min = min(d_min, np.sqrt(gx * gx + gy * gy))
                d_max = max(d_max, np.sqrt(gx * gx + gy * gy))
        scores = np.array(scores)

        for i in range(len(goals)):
            gx = goals[i][0]
            gy = goals[i][1]
            novel = True
            for old_gx, old_gy in replay_buffer:
                dx = old_gx - gx
                dy = old_gy - gy
                if np.sqrt(dx * dx + dy * dy) <= eps:
                    novel = False
                    break
            if novel:
                replay_buffer.append([gx, gy])
 

        logger.log("********** Iteration %i ************" % (it))
        logger.record_tabular("d_min", d_min)
        logger.record_tabular("d_max", d_max)
        logger.record_tabular("score_max", np.max(scores))
        logger.record_tabular("score_min", np.min(scores))
        logger.record_tabular("score_med", np.median(scores))
        logger.record_tabular("score_mean", np.mean(scores))
        logger.record_tabular("# of training goals:", goals.shape[0])
        coverage = 0
        for _ in range(100):
            gx = np.random.rand() * 5
            gy = np.random.rand() * 5
            _, reach = rollout(env, pi, [(gx, gy)], 1)
            if reach[0] != 0:
                coverage += 0.01
        logger.record_tabular("Coverage", coverage)
        logger.dump_tabular()
