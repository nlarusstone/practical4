import numpy.random as npr
import sys
import time
import matplotlib.pyplot as plt

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
        self.states = []

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.score = 0

    def disc_state(self, old_state):
        old_tree_bot = old_state['tree']['bot']
        new_tree_bot = (old_tree_bot - 11) / 20
        new_tree_bot = max(0, new_tree_bot)
        # new_tree_bot = min(6, new_tree_bot)
        
        old_tree_dist = old_state['tree']['dist']
        new_tree_dist = (old_tree_dist + 115) / 100
        new_tree_dist = max(0, new_tree_dist)
        # new_tree_dist = min(30, new_tree_dist)

        old_bot_diff = old_state['tree']['bot'] - old_state['monkey']['bot']
        new_bot_diff = (old_bot_diff + 300) / 100
        new_bot_diff = max(0, new_bot_diff)
        
        old_monk_vel = old_state['monkey']['vel']
        new_monk_vel = 0 if old_monk_vel < 0 else 1

        return (new_tree_bot, new_bot_diff, new_tree_dist, new_monk_vel)
        
        
    def action_callback(self, state):
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''
        
        self.states.append(state)

        if self.last_state is None:
            self.last_state = state
            self.last_action = 0
            return 0

        discount = 0.1

        # self.states.append(state)

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

        print len(self.Q)

        # CHOOSE NEW ACTION
        rnd = npr.random()

        if rnd > self.epsilon:
            # choose optimal action
            action_vals = self.Q[cur_state]
            new_action = 0 if action_vals[0] >= action_vals[1] else 1
        else:
            new_action = npr.random_integers(0, 1)

        self.last_action = new_action
        self.last_state  = state
        self.score = state['score']

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward

iters = 50
learner = Learner()
scores = []

for ii in xrange(iters):

    learner.epsilon = 1./(ii+1)

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

print len(learner.Q.keys())

tree_bots = [x['tree']['bot'] for x in learner.states]
diffs = [x['tree']['bot'] - x['monkey']['bot'] for x in learner.states]
vels = [x['monkey']['vel'] for x in learner.states]
dists = [x['tree']['dist'] for x in learner.states]

print 'tree bots'
print min(tree_bots)
print max(tree_bots)
print 'diffs'
print min(diffs)
print max(diffs)
print 'vels'
print min(vels)
print max(vels)
print 'dists'
print min(dists)
print max(dists)
