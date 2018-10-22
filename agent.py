from collections import deque

import gym
import numpy as np
import time
import random
from random_batch_deque import RandomBatchDeque

import tensorflow as tf
#from tensorflow import keras
import h5py


def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def downsample(img):
    return img[::2, ::2]


def preprocess(img):
    return to_grayscale(downsample(img))


def transform_reward(reward):
    return np.sign(reward)


class DQN:

    def __init__(self, state_size, action_size):
        self.sess = tf.Session()
        self.graph = tf.Graph()

        inference_variable = tf.Variable(np.ones((4, 105, 80, 1)), dtype=tf.int8, name="inference_variable")
        self.update_operation = inference_variable.assign

        self.queue = tf.FIFOQueue(capacity=150, dtypes=(tf.int8, tf.float32))
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

    def remember(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = tf.constant(np.array(state).reshape(4, 105, 80, 1), dtype=tf.int8)

        self.sess.run(self.update_operation(state))
        act_values = self.sess.run(self.output)
        #act_values = self.model.predict(state, steps=1)
        return np.argmax(act_values[0])  # returns action

    def fill_queue(self):

        batch_size = 1

        with tf.name_scope("queue_fill"):
            def inner_fill():
                ##while len(self.memory) < batch_size:
                 #   time.sleep(1)

                state, action, reward, done = random.sample(self.memory, batch_size)[0]
                target = reward
                if not done:
                    state_tbp = tf.slice(state, [1, 0, 0, 0], [4, 105, 80, 1])

                    self.sess.run(self.update_operation(state_tbp))

                    target = reward + self.gamma * np.amax(self.sess.run(self.output)[0])
                    #target = reward + self.gamma * np.amax(self.model.predict(state_tbp)[0])
                state_f = tf.slice(state, [0, 0, 0, 0], [4, 105, 80, 1])

                self.sess.run(self.update_operation(state_f))
                target_f = self.sess.run(self.output)
                #target_f = self.model.predict(state_f, steps=1)
                target_f[0][action] = target

                return self.queue.enqueue((state_f, tf.constant(target_f, dtype=tf.float32)))

            number_of_threads = 4
            qr = tf.train.QueueRunner(self.queue, [inner_fill()] * number_of_threads)
            tf.train.add_queue_runner(qr)

        return self.queue.dequeue()

    def replay(self, batch_size):

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('.')
        writer.add_graph(tf.get_default_graph())

        replay_start_t = int(round(time.time() * 1000))

        self.sess.run(self.model)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        print("ReplayE:" + str(int(round(time.time() * 1000)) - replay_start_t), flush=True)

    def input_fn(self):
        state, target = self.queue.dequeue()
        return state, target

    def generate_model(self):

        with tf.variable_scope('training_part', reuse=tf.AUTO_REUSE):

            input_t, true_t = self.queue.dequeue()
            input_t_cast = tf.cast(input_t, tf.float32)
            input_t_map = tf.map_fn(lambda x: x / 255.0, input_t_cast)
            input_t_reshaped = tf.reshape(input_t_map, (4, 105, 80, 1), name="reshape1")

            conv_layer_1 = tf.layers.conv2d(input_t_reshaped, 16, 8, 8, activation="relu", data_format="channels_last", name="conv2d_1", reuse=None)
            conv_layer_2 = tf.layers.conv2d(conv_layer_1, 32, 4, 4, activation="relu", name="conv2d_2", reuse=None)

            conv_flattened = tf.layers.Flatten(name="flatten1")(conv_layer_2)

            hidden = tf.layers.dense(conv_flattened, 256, activation="relu", name="hidden1", reuse=None)

            output = tf.layers.dense(hidden, self.action_size, name="output1", reuse=None)

            #########################################################################################

            input_i = tf.get_variable(name="inference_variable", shape=(4, 105, 80, 1))
            input_i_cast = tf.cast(input_i, tf.float32)
            input_i_map = tf.map_fn(lambda x: x / 255.0, input_i_cast)
            input_i_reshaped = tf.reshape(input_i_map, (4, 105, 80, 1), name="reshape2")

            conv_layer_3 = tf.layers.conv2d(input_i_reshaped, 16, 8, 8, activation="relu", data_format="channels_last", name="conv2d_1", reuse=True)
            conv_layer_4 = tf.layers.conv2d(conv_layer_3, 32, 4, 4, activation="relu", name="conv2d_2", reuse=True)

            conv_flattened_2 = tf.layers.Flatten(name="flatten1")(conv_layer_4)

            hidden_2 = tf.layers.dense(conv_flattened_2, 256, activation="relu", name="hidden1", reuse=True)

            output_2 = tf.layers.dense(hidden_2, self.action_size, name="output1", reuse=True)
            self.output = output_2

            #####################################################################################

            loss = tf.losses.mean_squared_error(labels=true_t, predictions=output)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon)   #TODO: Epsilon may be wrong

            train_op = optimizer.minimize(loss, name="train_op")
            a = 1

            #TODO: Create a model that enables to input variables and inference

            init = tf.global_variables_initializer()
            self.sess.run(init)

        return train_op

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class Game:

    # initialize the game and the needed constants
    def __init__(self):
        self.env = gym.make('BreakoutDeterministic-v4')
        self.env._max_episode_steps = None
        self.state_size = preprocess(self.env.reset()).shape + (1,)
        self.action_size = self.env.action_space.n
        self.recent_states = deque(maxlen=5)
        self.coord = tf.train.Coordinator()
        self.first_start = True
        self.batch_size = 1

        self.agent = DQN(self.state_size, self.action_size)

    #
    def play_game(self, n_episodes):
        for e in range(n_episodes):
            self.env.reset()
            self.recent_states = deque(maxlen=5)

            self.play_episode(n_episodes)

    def play_episode(self, n_max_frames, e, n_episodes):
        for time_t in range(n_max_frames):
            if time_t < 5:
                state, reward, done, _ = self.env.step(0)  # TODO: Make this random or interpolate somehow
                state = preprocess(state)
                self.recent_states.append(state)
            else:
                action = self.agent.act(np.array(self.recent_states)[1:].reshape(4, 105, 80, 1))
                state, reward, done, _ = self.env.step(action)
                reward = transform_reward(reward)
                state = preprocess(state)
                self.recent_states.append(state)
                self.agent.remember(tf.constant(np.array(self.recent_states).reshape(5, 105, 80, 1), dtype=tf.int8),
                                    action, reward, done)

            self.env.render()

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
