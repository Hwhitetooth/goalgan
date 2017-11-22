import gym


class Env(object):
    def __init__(self, env_name, target_x = 1E3, target_y = 0.0, eps = 0.5, timelimit = 500):
        self.env = gym.make(env_name)
        self.env.unwrapped.walk_target_x = target_x
        self.env.unwrapped.walk_target_y = target_y
        self.eps = eps
        self.timelimit = timelimit
        self.step_cnt = 0

    def reset(self, new_x = None, new_y = None):
        if new_x is not None:
            self.env.unwrapped.walk_target_x = new_x
        if new_y is not None:
            self.env.unwrapped.walk_target_y = new_y 
        self.step_cnt = 0
        return self.env.reset()

    def step(self, a):
        self.step_cnt += 1
        s, r, done, info = self.env.step(a)
        if self.env.unwrapped.walk_target_dist <= self.eps:
            r = 1.0
            done = True
        else:
            r = 0.0
        if self.step_cnt == self.timelimit:
            done = True
        return s, r, done, info

    @property
    def action_space(self):
        return self.env.action_space

    def render(self):
        return self.env.render()

    def seed(self, seed):
        return self.env.seed(seed)


class EnvL2(object):
    def __init__(self, env_name, target_x = 1E3, target_y = 0.0, eps = 0.5, timelimit = 500):
        self.env = gym.make(env_name)
        self.env.unwrapped.walk_target_x = target_x
        self.env.unwrapped.walk_target_y = target_y
        self.eps = eps
        self.timelimit = timelimit
        self.step_cnt = 0

    def reset(self, new_x = None, new_y = None):
        if new_x is not None:
            self.env.unwrapped.walk_target_x = new_x
        if new_y is not None:
            self.env.unwrapped.walk_target_y = new_y
        self.step_cnt = 0
        return self.env.reset()

    def step(self, a):
        self.step_cnt += 1
        old_dist = self.env.unwrapped.walk_target_dist
        s, r, done, info = self.env.step(a)
        if done:
            s = self.env.reset()
            done = False
        r = -self.env.unwrapped.walk_target_dist
        if self.env.unwrapped.walk_target_dist <= self.eps:
            done = True
        if self.step_cnt == self.timelimit:
            done = True
        return s, r, done, info

    @property
    def action_space(self):
        return self.env.action_space

    def render(self):
        return self.env.render()

    def seed(self, seed):
        return self.env.seed(seed)

'''
if __name__ == "__main__":
    env = Env("RoboschoolAnt-v1", 0.5, 0.5)
    env.reset()
    while True:
        a = env.env.action_space.sample()
        s, r, done, info = env.step(a)
        print(r)
        if done:
            env.reset()
'''
