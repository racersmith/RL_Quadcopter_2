import numpy as np
from ounoise import OUNoise


class University:
    def __init__(self, action_space, patience=50, sigma=0.2, theta=0.15):
        self.action_index = np.arange(action_space)
        self.action_set = set(self.action_index)
        self.patience = patience
        self.patience_count = 0
        self.best_reward = -np.inf
        
        self.assignment_index = 0
        self.assignments = [
            [[0,1,2,3]],
            [[0, 1], [2, 3]],
            [[0, 3], [1, 2]],
            [[0, 2], [1, 3]],
            [[1, 2, 3]],
            [[0, 2, 3]],
            [[0, 1, 3]],
            [[0, 1, 2]],
            [],
        ]
        self.sigma = sigma
        self.min_sigma = 0.01
        self.max_sigma = 2.0

        self.noise = OUNoise(size=action_space, mu=0.0, theta=theta, sigma=sigma)
        
    def classroom(self, action, groups):
        new_action = np.zeros_like(action)
        action_set = []
        noise = self.noise()
        noise_index = 0
        for group in groups:
            new_action[group] = np.mean(action[group]) + noise[noise_index]
            # new_action[group] = np.random.choice(action[group])
            action_set += group
            noise_index += 1
        action_set = set(action_set)
        for i in self.action_set - action_set:
            new_action[i] = action[i] + noise[noise_index]
            noise_index += 1
            
        return new_action
    
    def grade(self, reward):
        self.noise.reset()
        if reward > self.best_reward:
            self.patience_count = 0
            self.best_reward = reward
            self.noise.sigma = max(0.25 * self.noise.sigma, self.min_sigma)
        else:
            self.patience_count += 1
            sigma = 1.11 * self.noise.sigma
            if sigma > self.max_sigma:
                self.noise.sigma = self.min_sigma
            else:
                self.noise.sigma = sigma
            
        if self.patience_count > self.patience and self.assignment_index < len(self.assignments)-1:
            print("\tGraduating!!!!")

            self.noise.sigma = self.sigma
            self.patience_count = 0
            self.assignment_index += 1
            self.best_reward = -np.inf
        
    def assignment(self, action):
        assignment = self.assignments[self.assignment_index]
        new_action = self.classroom(action, assignment)
        return new_action