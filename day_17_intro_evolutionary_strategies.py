import gym
import matplotlib.pyplot as plt
import time
import seaborn as sns

env = gym.make('CartPole-v1')

def evaluate(W):
    # reset the environment, i.e. going back to beginning
    X = env.reset()
    env.render()
    
    # we run for 200 time steps, because 200 is the max reward we can get
    for t in range(1,201):
        # 0 means go left, 1 means go right
        action = 0 if W@X < 0 else 1
        X, reward, done, _ = env.step(action)
        if done:
            return t
    return t


evaluate([8.81054195, 5.82105525, 7.1435554,  7.99971522])