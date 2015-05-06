import numpy.random as npr
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import math

from SwingyMonkey import SwingyMonkey


class Learner:

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.Q = {}
        self.alphas = {}
        self.epsilon = 1
        self.score = 0
        self.raw_states = []
        self.disc_states = []

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.score = 0

    def disc_state(self, old_state):
        old_monk_bot = old_state['monkey']['bot']
        new_monk_bot = (old_monk_bot + 11) / 80
        new_monk_bot = max(0, new_monk_bot)

        old_bot_diff = old_state['tree']['bot'] - old_state['monkey']['bot']
        new_bot_diff = (old_bot_diff + 333) / 80
        new_bot_diff = max(0, new_bot_diff)
        
        old_tree_dist = old_state['tree']['dist']
        new_tree_dist = (old_tree_dist + 115) / 120
        new_tree_dist = max(0, new_tree_dist)

        old_monk_vel = old_state['monkey']['vel']
        new_monk_vel = (old_monk_vel + 40) / 20
        new_monk_vel = max(0, new_monk_vel)

        return (new_monk_bot, new_bot_diff, new_tree_dist, new_monk_vel)
        # return (new_bot_diff, new_tree_dist, new_monk_vel)
        
        
    def action_callback(self, state):
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''
        
        self.raw_states.append(state)
        self.disc_states.append(self.disc_state(state))

        if self.last_state is None:
            self.last_state = state
            self.last_action = 0
            return 0

        discount = 0.9

        # UPDATE Q
        last_state = self.disc_state(self.last_state)
        cur_state = self.disc_state(state)
        last_action = self.last_action
        last_reward = self.last_reward
        
        if last_state not in self.Q:
            self.Q[last_state] = [0., 0.]
            self.alphas[last_state] = [1., 1.]
        if cur_state not in self.Q:
            self.Q[cur_state] = [0., 0.]
            self.alphas[cur_state] = [1., 1.]

        old_val = self.Q[last_state][last_action]
        alpha = self.alphas[last_state][last_action]
        self.Q[last_state][last_action] = old_val + (1./alpha)*(last_reward + discount*max(self.Q[cur_state]) - old_val)

        self.alphas[last_state][last_action] += 1.
        
        # CHOOSE NEW ACTION
        rnd = npr.random()

        if rnd > self.epsilon:
            # choose optimal action
            action_vals = self.Q[cur_state]
            new_action = 0 if action_vals[0] >= action_vals[1] else 1
        else:
            # act randomly
            new_action = npr.randint(0, 1)

        self.last_action = new_action
        self.last_state  = state
        self.score = state['score']

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward

iters = 200
learner = Learner()
scores = []

for ii in xrange(iters):

    learner.epsilon = 1./(ii+1)
    #learner.epsilon = 1./math.exp(-ii/50)

    # Make a new monkey object.
    swing = SwingyMonkey(sound=False,            # Don't play sounds.
                         text="Epoch %d" % (ii), # Display the epoch on screen.
                         tick_length=1,          # Make game ticks super fast.
                         action_callback=learner.action_callback,
                         reward_callback=learner.reward_callback)
    # Loop until you hit something.
    while swing.game_loop():
        pass
    
    scores.append(learner.score)

    # Reset the state of the learner.
    learner.reset()

tree_bots = [x['tree']['bot'] for x in learner.raw_states]
monk_bots = [x['tree']['bot'] for x in learner.raw_states]
diffs = [x['tree']['bot'] - x['monkey']['bot'] for x in learner.raw_states]
vels = [x['monkey']['vel'] for x in learner.raw_states]
dists = [x['tree']['dist'] for x in learner.raw_states]

state0 = [x[0] for x in learner.disc_states]
state1 = [x[1] for x in learner.disc_states]
state2 = [x[2] for x in learner.disc_states]
state3 = [x[3] for x in learner.disc_states]

plt.hist(state0)
plt.show()
plt.hist(state1)
plt.show()
plt.hist(state2)
plt.show()
plt.hist(state3)
plt.show()

# print 'monk_bots'
# print min(monk_bots)
# print max(monk_bots)
# print 'diffs'
# print min(diffs)
# print max(diffs)
# print 'vels'
# plt.hist(vels)
# plt.show()
# print min(vels)
# print max(vels)
# print 'dists'
# print min(dists)
# print max(dists)

def moving_average(a, n=15):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

plt.plot(scores)
plt.show()
plt.plot(moving_average(scores))
plt.show()
plt.hist(scores)
plt.show()
print max(scores)
