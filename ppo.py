import numpy as np
import tensorflow as tf
import gym, roboschool
from policy import MLPPolicy, CNNPolicy
from utils import Dataset, MovingMeanVar
import logger
from collections import deque
import os
import time
from env_wrapper import Env

def rollout(env, pi, goals, episodes_per_goal = 1, render = False):
    s_batch, a_batch, r_batch, v_batch, done_batch = [], [], [], [], []
    ep_length = []
    ep_reward = []
    for gx, gy in goals:
        env.reset(gx, gy)
        for ep in range(episodes_per_goal):
            cur_ep_length = cur_ep_reward = 0
            s = env.reset()
            done = False
            while not done:
                a, v = pi.get_a_and_v(s[None])
                # FIXME: a bug here
                if np.any(np.isnan(a)):
                    a = env.action_space.sample()
                s_next, r, done, _ = env.step(np.array(a))
                if render:
                    env.render()
                s_batch.append(s)
                a_batch.append(a)
                r_batch.append(r)
                v_batch.append(v)
                done_batch.append(done)
                cur_ep_length += 1
                cur_ep_reward += r
                s = s_next
            ep_length.append(cur_ep_length)
            ep_reward.append(cur_ep_reward)
    return {"s": np.array(s_batch),
            "a": np.array(a_batch),
            "r": np.array(r_batch),
            "v": np.array(v_batch),
            "done": np.array(done_batch)}, ep_reward

def process_data(raw, gamma, lamda):
    s_batch, a_batch, r_batch, v_batch, done_batch = raw["s"], raw["a"], raw["r"], raw["v"], raw["done"]
    horizon = s_batch.shape[0]
    g_batch = np.zeros(horizon, "float32")
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
        horizon = 2048,
        batch_size = 64,
        epochs = 10,
        lr = 3E-4,
        max_steps = 1000000,
        clip_eps = 0.2, 
        entropy_coefficient = 0.0,
        logdir = "/tmp/ppo_log",
        save_freqency = 100,
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

    # Loss of entropy.
    entropy_loss = -tf.reduce_mean(pi.pi.entropy()) * entropy_coefficient

    total_loss = pi_loss + v_loss + entropy_loss
    losses = [total_loss, pi_loss, v_loss, entropy_loss]
    losses_names = ["all", "pi", "v", "entropy"]

    # Optimization.
    step = tf.placeholder(tf.int32)
    lr = tf.train.polynomial_decay(lr, step, max_steps, 0.0)
    opt = tf.train.AdamOptimizer(lr, epsilon = 1E-5)
    train_op = opt.minimize(total_loss, var_list = pi.vars)

    # Training.
    sess.run(tf.global_variables_initializer())

    def update_policy(env, goals, iterations = 5):
        for it in range(iterations):
            raw_data, _ = rollout(env, pi, goals, 1, render)
            data = Dataset(process_data(raw_data, gamma, lamda), batch_size)
            sess.run(update_old)
            for _ in range(epochs):
                for batch in data:
                    train_feed= {pi.inputs: batch["s"], pi_old.inputs: batch["s"], a_taken: batch["a"], \
                            adv_sample: batch["adv"], v_sample: batch["g"], step: it * horizon}
                    sess.run(train_op, train_feed)

    def eval_policy(env, goals):
        results = []
        for gx, gy in goals:
            _, ep_reward = rollout(env, pi, [(gx, gy)], 10)
            score = sum(ep_reward) / len(ep_reward)
            results.append((score, (gx, gy)))
        return results

    score_min, score_max = 0.6, 0.9
    env = Env(env_name)
    r_min, r_max = 0.1, 0.5
    replay_buffer = deque(maxlen = 50000)
    for _ in range(100):
        r = np.random.rand() * (r_max - r_min) + r_min
        gx = np.random.rand() * r
        gy = np.sqrt(r * r - gx * gx)
        replay_buffer.append((gx, gy))
    for it in range(10000):
        new_goals = []
        for _ in range(66):
            r = np.random.rand() * (r_max - r_min) + r_min
            gx = np.random.rand() * r
            gy = np.sqrt(r * r - gx * gx)
            new_goals.append((gx, gy))
        old_goals = []
        for _ in range(34):
            old_goals.append(replay_buffer[np.random.randint(len(replay_buffer))])
        goals = new_goals + old_goals
        update_policy(env, goals, 5)
        results = eval_policy(env, goals)
        new_r_min, new_r_max = 1E10, 0
        for score, (gx, gy) in results:
            if score >= score_min and score <= score_max:
                new_r_min = min(new_r_min, np.sqrt(gx * gx + gy * gy))
                new_r_max = max(new_r_max, np.sqrt(gx * gx + gy * gy))
        if new_r_min > new_r_max:
            print("WARNING: new_r_min > new_r_max")
        else:
            r_min, r_max = new_r_min, new_r_max * 1.1
        for gx, gy in new_goals:
            novel = True
            for old_gx, old_gy in replay_buffer:
                dx = old_gx - gx
                dy = old_gy - gy
                if np.sqrt(dx * dx + dy * dy) <= 0.5:
                    novel = False
                    break
            if novel:
                replay_buffer.append((gx, gy))
        logger.log("********** Iteration %i ************" % (it))
        logger.record_tabular("Rmin", r_min)
        logger.record_tabular("Rmax", r_max)
        logger.dump_tabular()
