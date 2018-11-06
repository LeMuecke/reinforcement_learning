import gym
from collections import deque
import numpy as np
import tensorflow as tf

from datetime import datetime, timedelta

from agent import DQN


def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def downsample(img):
    return img[::2, ::2]


def preprocess(img):
    return to_grayscale(downsample(img))


def transform_reward(reward):
    return np.sign(reward)


class Game:

    # initialize the game and the needed constants
    def __init__(self):
        self.env = gym.make('Pong-v0')  # pong or road runner
        self.env._max_episode_steps = None
        self.state_size = preprocess(self.env.reset()).shape + (1,)
        self.action_size = self.env.action_space.n
        self.recent_states = deque(maxlen=5)
        self.coord = tf.train.Coordinator()
        self.first_start = True
        self.batch_size = 1

        self.agent = DQN(self, self.state_size, self.action_size)

    def get_action(self):
        return self.agent.act(np.array(self.recent_states)[1:].reshape(4, 105, 80, 1))

    def play_game(self, n_episodes):
        for e in range(n_episodes):
            self.env.reset()
            self.recent_states = deque(maxlen=5)

            self.play_episode(500000, e, n_episodes)

    def play_episode(self, n_max_frames, e, n_episodes):
        time_for_fps = datetime.now()
        frames_this_second = 0
        for time_t in range(n_max_frames):
            if time_t < 5:
                state, reward, done, _ = self.env.step(0)  # TODO: Make this random or interpolate somehow
                state = preprocess(state)
                self.recent_states.append(state)
            else:
                action = self.get_action()
                state, reward, done, _ = self.env.step(action)
                reward = transform_reward(reward)
                state = preprocess(state)
                self.recent_states.append(state)
                self.agent.remember(tf.constant(np.array(self.recent_states).reshape(5, 105, 80, 1), dtype=tf.int8),
                                    action, reward, done)

            self.env.render()

            if datetime.now() - time_for_fps > timedelta(seconds=1):
                time_for_fps = datetime.now()
                print("Current fps: " + str(frames_this_second))
                frames_this_second = 0
            else:
                frames_this_second += 1

            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(e, n_episodes, time_t))
                break
            if len(self.agent.memory) > self.batch_size:
                if self.first_start:
                    self.agent.fill_queue()
                    threads = tf.train.start_queue_runners(coord=self.coord, sess=self.agent.sess)
                    self.first_start = False

                self.agent.replay(self.batch_size)


game = Game()
game.play_game(500)
