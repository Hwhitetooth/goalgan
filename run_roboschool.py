import tensorflow as tf
from policy import MLPPolicy
import os
import time
import ppo

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help = "Environment name", default = "RoboschoolAnt-v1")
    parser.add_argument("--seed", help = "Random seed", type = int, default = 0)
    parser.add_argument("--lr", help = "Initial learning rate", type = float, default = 3E-4)
    parser.add_argument("--clip_eps", help = "Clipping parameter", type = float, default = 0.2)
    parser.add_argument("--logdir", help = "Logging directory", default = "log")
    parser.add_argument("--max_iters", help = "Max training iterations", type = int, default = 300)
    parser.add_argument("--num_goals", help = "Number of goals per policy update", type = int, default = 100)
    parser.add_argument("--r_min", help = "Lower bound of proper goals range", type = float, default = 0.2)
    parser.add_argument("--r_max", help = "Upper bound of proper goals range", type = float, default = 0.8)
    parser.add_argument("--eps", help="Distance between two close goals", type=float, default=0.1)
    parser.add_argument("--render", help = "Render or not", type = bool, default = False)
    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    logdir = os.path.expanduser(args.logdir)
    logdir += "/" + args.env + "/" + 'rmin%g'%args.r_min + '_rmax%g'%args.r_max+'_'+'#goal%d'%args.num_goals+'eps%g'%args.eps+time.strftime("%Y_%m_%dT%H:%M:%S", time.localtime())
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    def mlp_policy(sess, input_shape, output_size, scope):
        return MLPPolicy(sess, input_shape, output_size, scope)
    ppo.train(args.env, sess, mlp_policy, logdir, lr = args.lr, clip_eps = args.clip_eps,
              max_iters = args.max_iters, eps=args.eps, num_goals = args.num_goals, r_min = args.r_min, r_max = args.r_max,
              render = args.render)

if __name__ == "__main__":
    main()
