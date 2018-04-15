import numpy as np

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(self.size)
        self.state = np.copy(self.mu)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self, decay=None, sigma_min=None):
        """Reset the internal state (= noise) to mean (mu)."""

        if decay is not None and sigma_min is not None:
            self.sigma = max((1-decay) * self.sigma, sigma_min)

        self.state = np.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state

    def __call__(self):
        return self.sample()

    def update_mu(self, target):
        self.mu = target