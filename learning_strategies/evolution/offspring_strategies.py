import math
from copy import deepcopy

import torch

from .abstracts import BaseOffspringStrategy


class simple_gaussian_offspring(BaseOffspringStrategy):
    def __init__(self, elite_ratio, init_sigma, sigma_decay, offspring_num):
        self.elite_ratio = elite_ratio
        self.init_sigma = init_sigma
        self.sigma_decay = sigma_decay
        self.parent_model = None
        self.offspring_num = offspring_num

        self.elite_num = max(int(offspring_num * elite_ratio), 1)
        self.mu = 0
        self.curr_sigma = self.init_sigma

        _weight_denominator = sum(
            math.log(offspring_num + 0.5) - math.log(i)
            for i in range(1, offspring_num + 1)
        )
        self.weight = [
            (math.log(offspring_num + 0.5) - math.log(i)) / _weight_denominator
            for i in range(1, offspring_num + 1)
        ]

    def get_parent_model(self):
        return self.parent_model

    def init_offspring(self, network, agent_ids):
        network.init_weights(0, 1e-7)
        parent = {}
        for agent_id in agent_ids:
            parent[agent_id] = network
        self.parent_model = parent
        return self.parent_model, (self.mu, self.curr_sigma), self.offspring_num

    def evaluate(self, results):
        results = sorted(results, key=lambda l: l[1], reverse=True)
        elite_group = results[: self.elite_num]
        parent, parent_reward = elite_group[0]
        with torch.no_grad():
            # best agent as parent
            for _, model in parent.items():
                for param in model.parameters():
                    param *= self.weight[0]
            # weighted sum elite
            for i, elite in enumerate(elite_group[1:]):
                for k, models in elite[0].items():
                    for param in model.parameters():
                        param *= self.weight[1 + i]
                for k in parent.keys():
                    for parent_p, elite_p in zip(
                        parent[k].parameters(), elite[0][k].parameters()
                    ):
                        parent_p += elite_p
        self.parent_model = parent
        self.curr_sigma *= self.sigma_decay
        return self.parent_model, parent_reward, (self.mu, self.curr_sigma)
