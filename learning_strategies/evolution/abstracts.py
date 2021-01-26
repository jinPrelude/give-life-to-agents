import os
from abc import *


class BaseESLoop(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, env, network, cpu_num):
        self.env = env
        self.network = network
        self.cpu_num = cpu_num

    @abstractmethod
    def run(self):
        pass


class BaseOffspringStrategy(metaclass=ABCMeta):
    @abstractmethod
    def get_parent_model(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


class BaseRolloutWorker(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, env):
        os.environ["MKL_NUM_THREADS"] = "1"
        self.env = env

    @abstractmethod
    def rollout(self):
        pass
