from collections import deque

import random
from tensorflow import QueueBase


class RandomBatchDeque(QueueBase):

    def __init__(self, capacity, dtypes, shapes=None, names=None, queue_ref=None):
        super().__init__(dtypes, shapes, names, queue_ref)
        self.memory = deque(maxlen=capacity)

    def enqueue(self, vals, name=None):
        self.memory.append(vals)

    def dequeue(self, name=None, batch_size=32):
        return random.sample(self.memory, batch_size)
