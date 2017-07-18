#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
from Config import Config
from Environment import Environment
import numpy as np
from random import randrange

class Replay(object):
    def __init__(self):
        self.capacity = Config.REPLAY_BUFFER_SIZE
        self.min_size = int(self.capacity / 5)
        self.states =  np.zeros((Config.REPLAY_BUFFER_SIZE,Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.STACKED_FRAMES), dtype=np.float32)
        self.actions = np.zeros((Config.REPLAY_BUFFER_SIZE,Environment().get_num_actions()), dtype=np.float32)
        self.rewards = np.zeros((Config.REPLAY_BUFFER_SIZE), dtype=np.float32)
        self.terminals = np.zeros((Config.REPLAY_BUFFER_SIZE), dtype=np.bool)
        self.size = 0
        self.index = 0

    def add_experience(self, state, action, reward, done):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.terminals[self.index] = done
        if self.size < self.capacity:
            self.size += 1
        self.index = (self.index + 1) % Config.REPLAY_BUFFER_SIZE

    def get_batch(self, batch_size):
        """
        Samples a batch of the specified size by selecting a random start/end point and returning
        the contained sequence (as opposed to sampling each state separately).

        Args:
            batch_size: Length of the sampled sequence.
        Returns: A dict containing states, rewards, terminals and internal states
        """

        if self.size < self.min_size:
            return {}

        end = (self.index - randrange(self.size - batch_size)) % self.capacity
        start = (end - batch_size) % self.capacity
        if start < end:
            indices = list(range(start, end))
        else:
            indices = list(range(start, self.capacity)) + list(range(0, end))

        print(indices)
        return dict(
            states=self.states.take(indices, axis=0),
            actions=self.actions.take(indices),
            rewards=self.rewards.take(indices),
            terminals=self.terminals.take(indices),
        )


if __name__ == '__main__':
    memories = Replay()
    num_actions = Environment().get_num_actions()
    #per-state case
    for i in range(10000):
        state = np.zeros((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.STACKED_FRAMES))
        action = np.zeros((num_actions,))
        reward = np.zeros((1,))
        done = np.zeros((1,),dtype=np.bool)

        memories.add_experience(state, action, reward, done)

        batch = memories.get_batch(Config.TIME_MAX)
        if batch:
            print('batch : ', batch['states'].shape)
        print(i, ": ", memories.size, memories.index)