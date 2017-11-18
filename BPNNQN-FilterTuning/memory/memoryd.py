# -*- coding: utf-8 -*-
"""

Memory stores filter tuning data and provides means work with it

@author: Jingfeng Yang 2016-03-09

"""
#TODO solve the issue of when we reach the limit of 1M samples, need to change add functions and solve the question of what to do on the border of new and old samples


import numpy as np
import random


class MemoryD:
    
    # size of the memory
    size = None
    
    # N x 401 matrix, where N is the number of tuning steps we want to keep
    states = None

    state_size = None
    
    state_nr = None
    
    #: list of size N, stores actions agent took
    actions = None
    
    #: list of size N, stores rewards agent received upon doing an action
    rewards = None

    #: list of size N, stores tuning filter internal time
    time = None

    #: global index counter
    count = None
    
    def __init__(self, n, size, nr):
        """
        Initialize memory structure 
        :type self: object
        @param n: the number ostate_sizef game steps we need to store
        """
        self.state_size = size
        self.state_nr = nr
        self.size = n
        self.states = np.zeros((n,self.state_size), dtype=np.float64)
        self.actions = np.zeros((n,), dtype=np.uint8)
        self.rewards = np.zeros((n,), dtype=np.float64)
        self.time = np.zeros((n,), dtype=np.uint32)
        self.count = -1
        
    def add_first(self, next_state):
        """
        When a new tuning start we add initial filter state to the memory
        @param next_state: 401 np.uint8 matrix
        """
        self.states[(self.count + 1) % self.size] = np.transpose(next_state)
        self.time[(self.count + 1) % self.size] = 0
        self.count += 1

    def add(self, action, reward, next_state):
        """
        During the tuning we add few thing to memory
        @param action: the action agent decided to do
        @param reward: the reward agent received for his action
        @param next_state: next state of the filter
        """
        self.actions[(self.count) % self.size] = action
        self.rewards[(self.count) % self.size] = reward
        self.time[(self.count + 1) % self.size] = self.time[(self.count) % self.size] + 1
        self.states[(self.count + 1) % self.size] = np.transpose(next_state)
        self.count += 1        
    
    def add_last(self):
        """
        When the tuning ends we fill memory for the current state with corresponding
        values. It is useful to think that at this time moment agent is looking at
        tuning finished
        """
        self.actions[(self.count) % self.size] = 100
        self.rewards[(self.count) % self.size] = 100

    def get_minibatch(self, size):
        """
        Take n Transitions from the memory.
        One transition includes (state, action, reward, state)
        Returns ndarray with 4 lists inside (at Tambet's request)
        @param size: size of the minibatch (in our case it should be 32)
        """
        prestates = np.zeros((size,self.state_size * self.state_nr), dtype = np.float64)
        actions = np.zeros((size), dtype=np.int16)
        rewards = np.zeros((size), dtype=np.float32)
        poststates = np.zeros((size,self.state_size * self.state_nr), dtype = np.float64)

        #: Pick random n indices and save dictionary if not terminal state
        j = 0
        while j < size:
            i = random.randint(0, np.min([self.count, self.size]) - 1)
            # we don't want to take frames that are just after our rewrite point
            while (i > (self.count % self.size) and (i-(self.count % self.size)) < 4) or self.actions[i] == 100:
                i = random.randint(0, np.min([self.count, self.size]) - 1)

            # add a state into the batch unless it is an endstate
            if self.actions[i] != 100:
#                prestates[j] = self.get_state(i)
                prestates[j] = self.states[i]
                actions[j] = self.actions[i]
                rewards[j] = self.rewards[i]
#                poststates[j] = self.get_state(i + 1)
                poststates[j] = self.states[i+1]
                j += 1
            else:
                print "We have a problem!! We selected an endstate!"

        return [prestates, actions, rewards, poststates]    

    def get_state(self, index):
        """
        Extract one state (4 waves) given last waves position
        @param index: global location of the 4th waves in the memory
        """

        #: We always need 4 waves to compose one state. In the beginning of the
        #  tuning (at time moments 0, 1, 2) we do not have enough waves in the memory
        #  for this particular tuning. So we came up with an ugly hack: duplicate the
        #  first available waves as many times as needed to fill missing ones.

        index = index % self.size
        pad_states = 3 - self.time[index]
        if pad_states > 0:

            state = np.zeros((self.state_nr,self.state_size), dtype=np.float64)

            #: Pad missing waves with the first wave
            for p in range(pad_states):
                state[p] = self.states[index - 3 + pad_states]

            #: Fill the rest of the waves as they are
            for p in range(pad_states, 4):
                state[p] = self.states[index - 3 + p]

        else:
            ind_start_window = index - 3
            ind_end_window = index + 1
            if ind_start_window < 0:
                state = np.vstack([self.states[ind_start_window:], self.states[:ind_end_window]])
            else:
                state = self.states[index - 3:index + 1]

        # neural network expects flat input and np.ravel does just that
        return np.ravel(state)

    def get_last_state(self):
        """
        Get last 4 waves from the memory. Those images will go as an input for
        the neural network
        """

        return self.get_state(self.count)    
    