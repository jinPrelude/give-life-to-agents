import os
from copy import deepcopy

import numpy as np
import ray
import torch

from .abstracts import BaseRolloutWorker


@ray.remote(num_cpus=1)
class RNNRolloutWorker(BaseRolloutWorker):
    def __init__(
        self, env, parent_id, noise_id, worker_id, offspring_num, eval_ep_num=10
    ):
        super().__init__(env)
        self.parent = parent_id
        self.noise = noise_id
        self.worker_id = worker_id
        self.eval_ep_num = eval_ep_num
        self.offspring_num = offspring_num

    def rollout(self):
        rewards = []
        for _ in range(self.offspring_num):
            total_reward = 0
            current_model = {}
            for key, model in self.parent.items():
                current_model[key] = deepcopy(model)
                current_model[key].add_noise(self.noise[0], self.noise[1])
            for _ in range(self.eval_ep_num):
                states = self.env.reset()
                hidden_states = {}
                done = False
                for k, model in current_model.items():
                    hidden_states[k] = model.init_hidden()
                while not done:
                    actions = {}
                    with torch.no_grad():
                        # ray.util.pdb.set_trace()
                        for k, model in current_model.items():
                            s = torch.from_numpy(
                                states[k]["state"][np.newaxis, ...]
                            ).float()
                            a, hidden_states[k] = model(s, hidden_states[k])
                            actions[k] = torch.argmax(a).detach().numpy()
                    states, r, done, info = self.env.step(actions)
                    # self.env.render()
                    total_reward += r
            rewards.append([current_model, total_reward / self.eval_ep_num])
        return rewards
