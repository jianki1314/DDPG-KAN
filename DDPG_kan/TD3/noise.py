import numpy as np

class OUActionNoise:
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None, decay=0.99):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.decay = decay  # 添加衰减系数
        self.reset()
 
    def __call__(self):
        # 生成噪声
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        # 调整噪声强度
        self.decay_noise()
        return x
 
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
    
    def decay_noise(self):
        """逐步减小噪声强度。"""
        self.sigma *= self.decay
