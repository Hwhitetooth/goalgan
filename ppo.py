import numpy as np
import tensorflow as tf
import gym, roboschool
from utils import Dataset
import logger
from collections import deque
import os
import time
from env_wrapper import Env


def rollout(env, pi, goals, episodes = None, timesteps = None, render = False):
    if episodes is None and timesteps is None:
        raise ValueError("Both episodes and timesteps are None.")
    if episodes is not None and timesteps is not None:
        raise ValueError("Both episodes and timesteps are not None.")
    s_batch, a_batch, r_batch, v_batch, done_batch = [], [], [], [], []
    score = []
    for gx, gy in goals:
        s = env.reset(gx, gy)
        step = 0
        ep = 0
        total_reward = 0
        while True:
            a, v = pi.get_a_and_v(s[None])
            # FIXME: a bug here
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
            if timesteps is not None and step == timesteps:
                break
        score.append(total_reward * np.sqrt(gx * gx + gy * gy) / step)
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
        gamma = 0.998,
        lamda = 0.995,
        batch_size = 64,
        epochs = 1,
        lr = 3E-4,
        clip_eps = 0.2,
        max_iters = 300,
        eps = 0.5,
        num_goals = 100,
        r_min = 0.01,
        r_max = 0.02,
        render = False):
    env = gym.make(env_name)
    pi = policy_fn(sess, env.observation_space.shape, env.action_space, "pi")
    pi_old = policy_fn(sess, env.observation_space.shape, env.action_space, "pi_old")
    update_old = [tf.assign(var_old, var_new) for var_old, var_new in zip(pi_old.vars, pi.vars)]

    # Loss of surrogate policy objective.
    a_taken = tf.placeholder(pi.a.dtype, pi.a.shape) # cache the actions chosen
    ratio = tf.exp(pi.pi.log_prob(a_taken) - pi_old.pi.log_prob(a_taken))
    adv_sample = tf.placeholder(tf.float32, [None])
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_sample, tf.clip_by_value(ratio, 1 - clip_eps, 1 + clip_eps) * adv_sample))

    # Loss of value function.
    v_sample = tf.placeholder(tf.float32, [None])
    v_loss = tf.reduce_mean(tf.square(v_sample - pi.v))

    total_loss = pi_loss + v_loss

    # Optimization.
    step = tf.placeholder(tf.int32)
    opt = tf.train.AdamOptimizer(lr, epsilon = 1E-5)
    train_op = opt.minimize(total_loss, var_list = pi.vars)

    # Training.
    sess.run(tf.global_variables_initializer())

    def update_policy(env, goals, iterations = 5):
        for it in range(iterations):
            raw_data, score = rollout(env, pi, goals, episodes = 1, render = render)
            data = Dataset(process_data(raw_data, gamma, lamda), batch_size)
            sess.run(update_old)
            for _ in range(epochs):
                for batch in data:
                    train_feed= {pi.inputs: batch["s"], pi_old.inputs: batch["s"], a_taken: batch["a"], \
                            adv_sample: batch["adv"], v_sample: batch["g"], step: 0}
                    sess.run(train_op, train_feed)
        return zip(score, goals)

    '''
    def eval_policy(env, goals):
        results = []
        for gx, gy in goals:
            _, score = rollout(env, pi, [(gx, gy)], timesteps = 500, render = render)
            results.append((score, (gx, gy)))
        return results
    '''

    env = Env(env_name, eps = eps)
    d_min, d_max = 0.5, 0.7
    replay_buffer = deque(maxlen = 100)
    for _ in range(1):
        d = np.random.rand() * (d_max - d_min) + d_min
        gx = np.random.rand() * d
        gy = np.sqrt(d * d - gx * gx)
        replay_buffer.append((gx, gy))

    for it in range(max_iters):
        goals = []
        for _ in range(num_goals * 2 // 3):
            d = np.random.rand() * (d_max - d_min) + d_min
            gx = np.random.rand() * d
            gy = np.sqrt(d * d - gx * gx)
            goals.append((gx, gy))
        while len(goals) < num_goals:
            goals.append(replay_buffer[np.random.randint(len(replay_buffer))])
        results = list(update_policy(env, goals, 5)) # This is slow!!!
        labels = [(int(score >= r_min and score <= r_max), (gx, gy)) for score, (gx, gy) in results]
        new_d_min, new_d_max = 1E10, 0
        scores = []
        for score, (gx, gy) in results:
            scores.append(score)
            if score >= r_min and score <= r_max:
                new_d_min = min(new_d_min, np.sqrt(gx * gx + gy * gy))
                new_d_max = max(new_d_max, np.sqrt(gx * gx + gy * gy))
        scores = np.array(scores)
        if new_d_min > new_d_max:
            print("WARNING: new_r_min > new_r_max")
        else:
            d_min, d_max = new_d_min, new_d_max * 1.1

        for gx, gy in goals:
            novel = True
            for old_gx, old_gy in replay_buffer:
                dx = old_gx - gx
                dy = old_gy - gy
                if np.sqrt(dx * dx + dy * dy) <= eps:
                    novel = False
                    break
            if novel:
                replay_buffer.append((gx, gy))

        logger.log("********** Iteration %i ************" % (it))
        logger.record_tabular("d_min", d_min)
        logger.record_tabular("d_max", d_max)
        logger.record_tabular("score_max", np.max(scores))
        logger.record_tabular("score_min", np.min(scores))
        logger.record_tabular("score_med", np.median(scores))
        logger.record_tabular("score_mean", np.mean(scores))
        if it % 1 == 0:
            coverage = 0
            for _ in range(100):
                d = np.random.rand() * 5
                gx = np.random.rand() * d
                gy = np.sqrt(d * d - gx * gx)
                _, reach = rollout(env, pi, [(gx, gy)], 1)
                if reach[0] != 0:
                    coverage += 0.01
            logger.record_tabular("Coverage", coverage)
        logger.dump_tabular()
