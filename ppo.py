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

def rollout(env, pi, horizon, episodes, render = False):
    s = env.reset()
    s_batch = np.array([s for _ in range(horizon)])
    a_batch = np.array([env.action_space.sample() for _ in range(horizon)])
    r_batch = np.zeros(horizon, "float32")
    v_batch = np.zeros(horizon, "float32")
    done_batch = np.array([False] * horizon)
    ep_length = []
    ep_reward = []
    cur_ep_length = cur_ep_reward = t = 0
    while True:
        a, v = pi.get_a_and_v(s[None])
        # FIXME: a bug here
        if np.any(np.isnan(a)):
            a = env.action_space.sample()
        s_next, r, done, _ = env.step(np.array(a))
        if render:
            env.render()
        s_batch[t] = s
        a_batch[t] = a
        r_batch[t] = r
        v_batch[t] = v
        done_batch[t] = done
        t += 1
        cur_ep_length += 1
        cur_ep_reward += r
        s = s_next
        if done:
            ep_length.append(cur_ep_length)
            ep_reward.append(cur_ep_reward)
            episodes -= 1
            if episodes == 0:
                v_next = 0
                yield {"s": s_batch, "a": a_batch, "r": r_batch, "v": v_batch, "done": done_batch, "v_next": v_next,
                        "ep_length": ep_length, "ep_reward": ep_reward}
                return
            cur_ep_length = cur_ep_reward = 0
            s = env.reset()
        if t == horizon:
            v_next = 0 if done else pi.get_v(s[None])
            yield {"s": s_batch, "a": a_batch, "r": r_batch, "v": v_batch, "done": done_batch, "v_next": v_next,
                    "ep_length": ep_length, "ep_reward": ep_reward}
            t = 0
            ep_length = []
            ep_reward = []

def process_data(raw, horizon, gamma, lamda):
    s_batch, a_batch, r_batch, v_batch, done_batch, v_next  = raw["s"], raw["a"], raw["r"], raw["v"], raw["done"], raw["v_next"]
    g_batch = np.zeros(horizon, "float32")
    adv_batch = np.zeros(horizon, "float32")
    adv_next = 0
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
    data.pop("v_next")
    data["g"] = g_batch
    data["adv"] = adv_batch
    return data

def train(env_name,
        sess,
        policy_fn,
        gamma = 0.99,
        lamda = 0.95,
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
    saver = tf.train.Saver(max_to_keep = max_steps // (horizon * save_freqency) + 1)

    # Summary.
    summary_reward = tf.placeholder(tf.float32)
    tf.summary.scalar("Average episode reward", summary_reward)
    summary_length = tf.placeholder(tf.float32)
    tf.summary.scalar("Average episode length", summary_length)
    '''
    summary_losses = dict()
    for name, loss in zip(losses_names, losses):
        summary_losses[name] = tf.placeholder(tf.float32)
        tf.summary.scalar("loss_" + name, summary_losses[name])
    tf.summary.scalar("Learning rate", lr)
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logdir, graph = sess.graph)
    '''

    # Training.
    sess.run(tf.global_variables_initializer())

    def update_policy(env, targets_buffer, episodes = 10):
        targets = np.random.choice(len(targets_buffer), 10)
        ep = 0
        for i in targets:
            tx, ty = targets_buffer[i]
            env.reset(tx, ty)
            data_gen = rollout(env, pi, horizon, episodes, render)
            for raw_data in data_gen:
                raw_data.pop("ep_length")
                raw_data.pop("ep_reward")
                data = Dataset(process_data(raw_data, horizon, gamma, lamda), batch_size)
                sess.run(update_old)
                for _ in range(epochs):
                    for batch in data:
                        train_feed= {pi.inputs: batch["s"], pi_old.inputs: batch["s"], a_taken: batch["a"], \
                                adv_sample: batch["adv"], v_sample: batch["g"], step: it * horizon}
                        sess.run(train_op, train_feed)
            ep += 1
            print(ep)
        '''
        stats_length = deque(maxlen = 100)
        stats_reward = deque(maxlen = 100)
        stats_length.extend(raw_data.pop("ep_length"))
        stats_reward.extend(raw_data.pop("ep_reward"))
        '''

    def eval_policy(env, targets):
        stats_length = deque(maxlen = len(targets))
        stats_reward = deque(maxlen = len(targets))
        for tx, ty in targets:
            env.reset(tx, ty)
            data_gen = rollout(env, pi, horizon, 1, render)
            for raw_data in data_gen:
                stats_length.extend(raw_data.pop("ep_length"))
                stats_reward.extend(raw_data.pop("ep_reward"))
        mean_length = np.mean(np.array(stats_length), axis = 0)
        mean_reward = np.mean(np.array(stats_reward), axis = 0)
        return mean_length, mean_reward

    radius = 0.1
    env = Env(env_name)
    targets = deque(maxlen = 100)
    for i in range(5):
        x = np.random.rand() * radius
        y = np.sqrt(radius * radius - x * x)
        targets.append((x, y))
    for it in range(10000):
        update_policy(env, targets, 10)
        mean_length, mean_reward = eval_policy(env, targets)
        logger.log("********** Iteration %i ************" % (it))
        logger.record_tabular("Distance", radius)
        logger.record_tabular("Mean length", mean_length)
        logger.record_tabular("Mean reward", mean_reward)
        logger.dump_tabular()
        if mean_reward < 0.6:
            radius *= 0.8
        elif mean_reward > 0.99:
            radius *= 1.2
            for i in range(5):
                x = np.random.rand() * radius
                y = np.sqrt(radius * radius - x * x)
                targets.append((x, y))

    '''
    stats_losses= []
    for raw_data in data_gen:
        stats_length.extend(raw_data.pop("ep_length"))
        stats_reward.extend(raw_data.pop("ep_reward"))
        for batch in data:
            feed_dict = {pi.inputs: batch["s"], pi_old.inputs: batch["s"], a_taken: batch["a"], adv_sample: batch["adv"], v_sample: batch["g"]}
            stats_losses.append(sess.run(losses, feed_dict))
    summary_feed = dict()
    mean_losses = np.mean(np.array(stats_losses), axis = 0)
    for name, loss in zip(losses_names, mean_losses):
        logger.record_tabular("loss_" + name, loss)
        summary_feed[summary_losses[name]] = loss
    mean_length = np.mean(np.array(stats_length), axis = 0)
    summary_feed[summary_length] = mean_length
    mean_reward = np.mean(np.array(stats_reward), axis = 0)
    summary_feed[summary_reward] = mean_reward
    summary_feed[step] = it * horizon
    summary_str = sess.run(summary_op, feed_dict = summary_feed)
    summary_writer.add_summary(summary_str, it * horizon)
    saver.save(sess, logdir + "/ppo", it * horizon)
    '''
