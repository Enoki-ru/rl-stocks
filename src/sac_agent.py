import torch
import torch.optim as optim
import torch.nn.functional as F
from networks import TransformerFeatureExtractor, Actor, Critic
import copy

class SACAgent:
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        
        # --- Параметры из конфига ---
        n_comps = config['model']['n_comps']
        # Размерность признаков для одного актива (60 дней * 1 цена = 60)
        # Плюс информация о портфеле (вес, pnl) -> 60 + 2 = 62
        # Это нужно будет согласовать с вашей средой!
        n_features = config['model']['n_days_history'] + 2 
        d_model = config['model']['d_model']
        n_head = config['model']['n_head']
        num_layers = config['model']['num_layers']
        
        # Размерность пространства действий = кол-во компаний + кэш
        action_dim = n_comps + 1

        # --- Инициализация сетей ---
        # Общий извлекатель признаков
        self.feature_extractor = TransformerFeatureExtractor(n_comps, n_features, d_model, n_head, num_layers).to(device)
        
        # Актер
        self.actor = Actor(d_model, action_dim).to(device)
        
        # Критики (две штуки для стабильности)
        self.critic1 = Critic(d_model, action_dim).to(device)
        self.critic2 = Critic(d_model, action_dim).to(device)
        
        # Целевые сети для критиков (копируем архитектуру и веса)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        
        # --- Оптимизаторы ---
        # Собираем параметры всех сетей для совместного обучения
        all_params = list(self.feature_extractor.parameters()) + \
                     list(self.actor.parameters()) + \
                     list(self.critic1.parameters()) + \
                     list(self.critic2.parameters())
        self.optimizer = optim.Adam(all_params, lr=3e-4)

        # Отдельный оптимизатор для альфы (коэффициент энтропии)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -torch.prod(torch.Tensor((action_dim,)).to(device)).item()

        # --- Гиперпараметры SAC ---
        self.gamma = 0.99  # Коэффициент дисконтирования
        self.tau = 0.005   # Коэффициент для "мягкого" обновления целевых сетей

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Выбор действия для одного состояния (во время игры)"""
        state = state.unsqueeze(0).to(self.device) # Добавляем batch_dim
        with torch.no_grad():
            latent_state = self.feature_extractor(state)
            action, _ = self.actor.sample(latent_state)
        return action.cpu().squeeze(0) # Возвращаем на CPU без batch_dim

    def update(self, replay_buffer, batch_size: int):
        """Одна итерация обучения на батче из буфера"""
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # --- Перемещаем все данные на GPU ---
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        # --- Вычисление потерь для Критиков (Шаг 1 в таблице Backward) ---
        with torch.no_grad():
            # Получаем эмбеддинги для следующего состояния
            next_latent_state = self.feature_extractor(next_state)
            # Получаем следующие действия и их лог-вероятности от Актера
            next_action, next_log_pi = self.actor.sample(next_latent_state)
            
            # Вычисляем Q-value от целевых критиков
            q1_target_next = self.critic1_target(next_latent_state, next_action)
            q2_target_next = self.critic2_target(next_latent_state, next_action)
            # Берем минимум для борьбы с завышением оценок
            min_q_target_next = torch.min(q1_target_next, q2_target_next)
            
            # Вычисляем целевое Q-value с учетом энтропии
            alpha = self.log_alpha.exp()
            q_target = reward + self.gamma * (1 - done) * (min_q_target_next - alpha * next_log_pi)

        # Текущие Q-value от основных критиков
        latent_state = self.feature_extractor(state)
        q1_current = self.critic1(latent_state, action)
        q2_current = self.critic2(latent_state, action)
        
        # Loss - это MSE между текущим и целевым Q-value
        critic1_loss = F.mse_loss(q1_current, q_target)
        critic2_loss = F.mse_loss(q2_current, q_target)
        critic_loss = critic1_loss + critic2_loss

        # --- Вычисление потерь для Актера (Шаг 2) ---
        # Замораживаем градиенты для критиков, чтобы оптимизировать только актера
        # (хотя оптимизатор у нас общий, но это хорошая практика)
        pi, log_pi = self.actor.sample(latent_state)
        q1_pi = self.critic1(latent_state, pi)
        q2_pi = self.critic2(latent_state, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = ((alpha * log_pi) - min_q_pi).mean()

        # --- Вычисление потерь для Alpha (Шаг 3) ---
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        # --- Обновление весов (Шаг 4) ---
        # Обновляем все сети (Transformer, Actor, Critic) одним махом
        self.optimizer.zero_grad()
        # Суммируем потери и делаем backward pass
        total_loss = critic_loss + actor_loss
        total_loss.backward()
        self.optimizer.step()

        # Обновляем alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- "Мягкое" обновление целевых сетей ---
        with torch.no_grad():
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)