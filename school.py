import numpy as np
# from ounoise import OUNoise


class FlightShool:
    def __init__(self, action_space, patience=50, sigma=0.2, theta=0.15):
        self.action_index = np.arange(action_space)
        self.action_set = set(self.action_index)
        self.patience = patience
        self.patience_count = 0
        self.best_score = -np.inf
        self.worst_score = np.inf
        
        self.curriculum_index = 0
        self.curriculum = [
            [[0, 1, 2, 3]],  # Elevation
            [[0, 2], [1, 3]],  # Yaw, about z
            [[2, 3], [0, 1]],  # Pitch, about y
            [[0, 3], [1, 2]],  # Roll, about x

            # [[1, 2, 3]], # Single rotor
            # [[0, 2, 3]],
            # [[0, 1, 3]],
            # [[0, 1, 2]],
            [],  # Free
        ]
        # self.sigma = sigma
        # self.min_sigma = 0.01
        # self.max_sigma = 2.0

        self.soft_score = None
        self.gamma = 0.98

        self.review_rate = 0.10

        # Create a initial grade set that forces progression
        # increasing the high will slow graduating to next class
        # self.grades = -np.linspace(50, 10, len(self.curriculum))**2
        # self.grades = -np.logspace(np.log10(250), np.log10(100), len(self.curriculum))
        # self.grades = np.array([-250, -175, -175, -175, -150])
        # self.grades = np.array([-250, -250, -250, -250, -225])
        self.grades = np.ones(len(self.curriculum))

        # self.noise = OUNoise(size=action_space, mu=0.0, theta=theta, sigma=sigma)


        
    def classroom(self, action, groups):
        new_action = np.zeros_like(action)
        action_set = []
        # noise = self.noise()
        noise_index = 0
        for group in groups:
            new_action[group] = np.mean(action[group])
            # new_action[group] = np.random.choice(action[group])
            # new_action[group] += noise[noise_index]
            action_set += group
            noise_index += 1
        action_set = set(action_set)
        for i in self.action_set - action_set:
            # new_action[i] = action[i] + noise[noise_index]
            noise_index += 1
            
        return new_action

    # def update_soft_score(self, score):
    #     if self.soft_score is None:
    #         # Some of the first passes are actually quite good
    #         # Don't get your hopes up at this point
    #         self.soft_score = min(score*0.2, score*5)
    #     else:
    #         self.soft_score = self.gamma*self.soft_score + (1-self.gamma)*score

    def update_score(self, score):
        self.worst_score = min(self.worst_score, score)
        self.best_score = max(self.best_score, score)

    # def adapt_noise(self, score):
    #     if score > self.best_score:
    #         self.noise.sigma = max(0.25 * self.noise.sigma, self.min_sigma)
    #     else:
    #         self.noise.sigma = min(1.1 * self.noise.sigma, self.max_sigma)

    # def update_patience(self, score):
    #     if 0.95*score >= self.best_score:
    #         self.patience_count = 0
    #     else:
    #         self.patience_count += 1

    def epsilon_greedy(self, epsilon):
        policy = []
        n = len(self.grades)
        policy = np.ones(n) * epsilon / n
        policy[np.argmin(self.grades)] += 1 - epsilon
        return policy

    def ranked_policy(self, epsilon):
        alpha = len(self.grades)/(1+epsilon)
        p = np.exp(np.linspace(alpha, 0, len(self.grades), endpoint=True))
        p = p / np.sum(p)
        return p[np.argsort(self.grades)]

    def weighted_policy(self, epsilon):
        g_max = np.max(self.grades)
        g_min = np.min(self.grades)
        p = g_max - self.grades + epsilon * (g_max - g_min)
        p = p**2
        return p/np.sum(p)

    def update_p(self, score):
        # self.p = np.zeros(len(self.curriculum))
        # self.p[:max(1, self.curriculum_index)] = self.review_rate/ max(1, self.curriculum_index)
        # self.p[self.curriculum_index] += 1 - self.review_rate
        self.grades[self.curriculum_index] = self.soft_update(self.grades[self.curriculum_index], score, self.gamma)
        # self.p = self.epsilon_greedy(self.review_rate)
        # self.p = self.ranked_policy(self.review_rate)
        self.p = self.weighted_policy(self.review_rate)

    def grade(self, score):
        # self.noise.reset()
        # self.update_soft_score(score)
        # self.update_patience(score)
        # self.update_score(self.soft_score)
        # self.update_p(score)

        self.grades[self.curriculum_index] = self.soft_update(self.grades[self.curriculum_index], score, self.gamma)
        # self.curriculum_index = np.random.choice(np.arange(len(self.curriculum)), p=self.p)

        # if self.patience_count > self.patience and self.curriculum_index < len(self.curriculum)-1:
        #     print("  Graduating!!!!")
        #
        #     self.noise.sigma = self.sigma
        #     self.patience_count = 0
        #     self.curriculum_index += 1
        #     self.best_score = -np.inf
        #     self.soft_score = self.worst_score
        #     self.update_p()

    def assign_lesson(self, lesson_id=None):
        # self.curriculum_index = np.random.choice(np.arange(len(self.curriculum)), p=self.p)
        if lesson_id is None:
            self.curriculum_index = (self.curriculum_index + 1) % len(self.curriculum)
        else:
            if lesson_id >= 0:
                self.curriculum_index = lesson_id % len(self.curriculum)
            else:
                self.curriculum_index = len(self.curriculum)-1
        # self.curriculum_index = np.random.choice(np.arange(len(self.curriculum)))

    def soft_update(self, old, new, gamma):
        if old is None:
            return old

        return gamma*old + (1-gamma)*new
        
    def lesson(self, action):
        new_action = self.classroom(action, self.curriculum[self.curriculum_index])
        return new_action