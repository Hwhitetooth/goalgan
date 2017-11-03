import numpy as np
import tensorflow as tf
import gym, roboschool
from policy import MLPPolicy
import os
import time
import ppo

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help = "Environment name", default = "RoboschoolAnt-v1")
    parser.add_argument("--seed", help = "Random seed", type = int, default = 0)
    parser.add_argument("--render", help = "Render or not", type = bool, default = False)
    parser.add_argument("--max_steps", help = "Max training steps", type = int, default = 1000000)
    parser.add_argument("--lr", help = "Initial learning rate", type = float, default = 3E-4)
    parser.add_argument("--clip_eps", help = "Clipping parameter", type = float, default = 0.2)
    parser.add_argument("--logdir", help = "Logging directory", default = "~/log")
    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)

    logdir = os.path.expanduser(args.logdir)
    logdir += "/" + args.env + "/" + time.strftime("%Y_%m_%dT%H:%M:%S", time.localtime())
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    def mlp_policy(sess, input_shape, output_size, scope):
        return MLPPolicy(sess, input_shape, output_size, scope)
    ppo.train(args.env, sess, mlp_policy, lr = args.lr, max_steps = args.max_steps, clip_eps = args.clip_eps, render = args.render, logdir = logdir)

if __name__ == "__main__":
    main()
