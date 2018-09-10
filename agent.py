from collections import deque

import gym
import numpy as np
import time
import random
from random_batch_deque import RandomBatchDeque

import tensorflow as tf
from tensorflow import keras
import h5py


class DQN():

    def __init__(self, state_size, action_size):
        self.sess = tf.Session()
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.00025
        self.rho = 0.95
        self.model = self.generate_model()
        self.queue = RandomBatchDeque(capacity=2000, dtypes=np.uint8)

    def remember(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = tf.slice(state, [0, 0, 0, 0], [4, 1, 105, 80])
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        replay_start_t = int(round(time.time() * 1000))
        #minibatch = random.sample(self.memory, batch_size)
        minibatch = self.queue.dequeue(batch_size=batch_size)

        for state, action, reward, done in minibatch:
            target = reward
            if not done:
                states_to_be_predicted = tf.slice(state, [1, 0, 0, 0], [5, 1, 105, 80])
                #states_to_be_predicted = state[1:5].reshape(4, 1, 105, 80)

                target = reward + self.gamma * np.amax(self.model.predict(states_to_be_predicted)[0])
            state_f = tf.slice(state, [0, 0, 0, 0], [4, 1, 105, 80])
            target_f = self.model.predict(state_f)
            target_f[0][action.eval(session=self.sess)] = target    #TODO: Has to be converted to tensor
            self.model.fit(state_f, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        print("Replay " + str(int(round(time.time() * 1000)) - replay_start_t), end="", flush=True)

    def generate_model(self):

        input_layer = keras.layers.Input(shape=self.state_size, batch_size=4, name='input_frames')

        normalized = keras.layers.Lambda(lambda x: x / 255.0)(input_layer)

        conv_layer_1 = keras.layers.Conv2D(16, 8, 8, activation="relu", data_format="channels_first")(normalized)
        conv_layer_2 = keras.layers.Conv2D(32, 4, 4, activation="relu")(conv_layer_1)

        conv_flattened = keras.layers.Flatten()(conv_layer_2)

        hidden = keras.layers.Dense(256, activation="relu")(conv_flattened)

        output = keras.layers.Dense(self.action_size)(hidden)

        model = keras.models.Model(inputs=input_layer, outputs=output)
        optimizer = keras.optimizers.RMSprop(lr=self.learning_rate, rho=self.rho, epsilon=self.epsilon)
        model.compile(optimizer, loss='mse')
        return model

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def downsample(img):
    return img[::2, ::2]


def preprocess(img):
    return to_grayscale(downsample(img))


def transform_reward(reward):
    return np.sign(reward)


def train(episodes):
    #env = gym.make('CartPole-v0')
    #env = gym.make('Breakout-v0')
    #env = gym.make('BeamRider-v0')
    env = gym.make('BreakoutDeterministic-v4')

    env._max_episode_steps = None
    state_size = (1,) + preprocess(env.reset()).shape
    action_size = env.action_space.n
    agent = DQN(state_size, action_size)

    done = False
    batch_size = 32

    #agent.load("./breakoutDeterministicV4.h5")
    try:
        for e in range(episodes):
            episode_start_t = int(round(time.time() * 1000))
            state = preprocess(env.reset())
            action = None

            for time_t in range(100000):

                frame_collector = list()
                #reward_collector = list()
                if time_t == 0:
                    frame_collector.append(state)
                else:
                    action = tf.constant(agent.act(state), dtype=tf.int8)
                    next_state, reward, done, _ = env.step(action.eval(session=agent.sess))
                    reward = tf.constant(transform_reward(reward))
                    next_state = preprocess(next_state)
                    state.append(next_state)

                    #agent.remember(tf.constant(np.array(state).reshape(5, 1, 105, 80), dtype=tf.int8),
                    #               action, reward, tf.constant(done, dtype=tf.int8))
                    agent.queue.enqueue((tf.constant(np.array(state).reshape(5, 1, 105, 80), dtype=tf.int8),
                                         action, reward, tf.constant(done, dtype=tf.int8)))

                    frame_collector.append(next_state)
                for frame in range(3):
                    next_state, reward, done, _ = env.step(0)   #TODO: Check if 0 is really "not moving anywhere"
                    frame_collector.append(preprocess(next_state))
                    #reward_collector.append(transform_reward(reward))

                state = frame_collector.copy()

                #env.render()

                if done:
                    # print the score and break out of the loop
                    print("episode: {}/{}, score: {}"
                          .format(e, episodes, time_t))
                    break
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

            print("Episode took " + str(int(round(time.time() * 1000)) - episode_start_t))

        agent.save("./breakoutDeterministicV4.h5")
    except KeyboardInterrupt:
        agent.save("./breakoutDeterministicV4.h5")


train(50000)
