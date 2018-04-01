import numpy as np


class University:
    def __init__(self, action_space, graduation=50):
        self.action_index = np.arange(action_space)
        self.action_set = set(self.action_index)
        self.graduation = graduation
        self.improve_count = 0
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
        
    def classroom(self, action, groups):
        new_action = np.zeros_like(action)
        action_set = []
        for group in groups:
            new_action[group] = np.mean(action[group])
            # new_action[group] = np.random.choice(action[group])
            action_set += group
        action_set = set(action_set)
        for i in self.action_set - action_set:
            new_action[i] = action[i]
            
        return new_action
    
    def grade(self, reward):
        if reward > self.best_reward:
            self.improve_count = 0
            self.best_reward = reward
        else:
            self.improve_count += 1
            self.best_reward = -np.inf
            
        if self.improve_count > self.graduation and self.assignment_index < len(self.assignments)-1:
            self.assignment_index += 1
            self.best_reward = -np.inf
        
    def assignment(self, action):
        assignment = self.assignments[self.assignment_index]
        new_action = self.classroom(action, assignment)
        return new_action