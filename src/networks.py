import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math

class PositionalEncoding(nn.Module):
    """
    Классическая позиционная кодировка из статьи "Attention Is All You Need".
    Она добавляет информацию о позиции токена (в нашем случае - актива) в последовательности.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor, shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerFeatureExtractor(nn.Module):
    """
    Этот модуль является "сенсорной корой" нашего агента.
    Он принимает "сырые" данные по всем активам и извлекает из них "суть"
    в виде контекстуализированных эмбеддингов.
    """
    def __init__(self, n_comps: int, n_features: int, d_model: int, n_head: int, num_layers: int):
        super().__init__()
        self.d_model = d_model
        
        # 1. Входной слой: преобразует вектор признаков каждого актива в d_model
        self.input_projection = nn.Linear(n_features, d_model)
        
        # 2. Позиционная кодировка: сообщает модели о порядке активов (если он важен)
        # Для портфеля можно использовать и обучаемые эмбеддинги, но классика тоже работает.
        self.pos_encoder = PositionalEncoding(d_model, max_len=n_comps)
        
        # 3. Сам Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=d_model*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, n_comps, n_features] - батч состояний
        
        # Проекция в пространство d_model
        x = self.input_projection(x) # -> [batch_size, n_comps, d_model]
        
        # Добавление позиционной информации
        # TransformerEncoder в PyTorch по умолчанию ожидает [seq_len, batch_size, d_model]
        x = x.permute(1, 0, 2) # -> [n_comps, batch_size, d_model]
        x = self.pos_encoder(x)
        
        # Прогон через энкодер
        output = self.transformer_encoder(x) # -> [n_comps, batch_size, d_model]
        
        # Возвращаем к формату batch_first
        output = output.permute(1, 0, 2) # -> [batch_size, n_comps, d_model]
        
        return output

class Actor(nn.Module):
    """
    Сеть "Актёра" (Policy Network). Принимает обработанное состояние и решает, что делать.
    """
    def __init__(self, d_model: int, action_dim: int):
        super().__init__()
        # Простая MLP для принятия решения на основе эмбеддингов
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU()
        )
        # "Головы" для вычисления параметров распределения действий
        self.mu_head = nn.Linear(d_model // 4, action_dim)
        self.log_std_head = nn.Linear(d_model // 4, action_dim)

    def forward(self, latent_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # latent_state: [batch_size, n_comps, d_model]
        
        # Агрегируем информацию по всем активам в один вектор (например, средним)
        # Это позволяет модели принимать одно глобальное решение по портфелю
        aggregated_state = latent_state.mean(dim=1) # -> [batch_size, d_model]
        
        x = self.net(aggregated_state)
        
        # Вычисляем параметры для Гауссова распределения
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        # Ограничиваем log_std для стабильности обучения
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mu, log_std

    def sample(self, latent_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Реализует сэмплирование с reparameterization trick для возможности backprop.
        """
        mu, log_std = self.forward(latent_state)
        std = log_std.exp()
        normal = Normal(mu, std)
        x_t = normal.rsample()  # rsample() = reparameterization trick
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        
        # Коррекция log_prob из-за использования tanh
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    """
    Сеть "Критика" (Q-Network). Оценивает, насколько хорошо действие в данном состоянии.
    """
    def __init__(self, d_model: int, action_dim: int):
        super().__init__()
        # Критик принимает на вход и состояние, и действие
        self.net = nn.Sequential(
            nn.Linear(d_model + action_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1) # Выход - одно число (Q-value)
        )

    def forward(self, latent_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # latent_state: [batch_size, n_comps, d_model]
        # action: [batch_size, action_dim]
        
        # Агрегируем состояние так же, как в Актёре
        aggregated_state = latent_state.mean(dim=1) # -> [batch_size, d_model]
        
        # Объединяем состояние и действие
        x = torch.cat([aggregated_state, action], dim=1)
        
        return self.net(x)
