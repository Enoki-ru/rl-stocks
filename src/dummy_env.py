import gymnasium as gym
import numpy as np

class DummyTradingEnv(gym.Env):
    """
    Простейшая имитация среды. Возвращает случайные данные.
    Вам нужно будет заменить это на вашу реальную среду с историческими данными.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.n_comps = config['model']['n_comps']
        # 60 дней истории + 2 признака портфеля (вес, pnl)
        self.n_features = config['model']['n_days_history'] + 2
        self.action_dim = self.n_comps + 1
        
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_comps, self.n_features), dtype=np.float32)
        
        self.step_count = 0

    def reset(self, seed=None, options=None):
        self.step_count = 0
        # Возвращаем случайное начальное состояние
        obs = np.random.rand(self.n_comps, self.n_features).astype(np.float32)
        return obs, {}

    def step(self, action):
        self.step_count += 1
        # Имитируем шаг: следующее состояние - случайное, награда - случайная
        next_obs = np.random.rand(self.n_comps, self.n_features).astype(np.float32)
        reward = np.random.rand() * 2 - 1 # Случайная награда от -1 до 1
        
        # Завершаем эпизод после 200 шагов
        done = self.step_count >= 200
        truncated = False
        
        return next_obs, reward, done, truncated, {}