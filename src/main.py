import yaml
import torch
from sac_agent import SACAgent
from replay_buffer import ReplayBuffer
from dummy_env import DummyTradingEnv

# --- Использование GPU vs CPU ---
# PyTorch позволяет легко переключаться между устройствами.
# 1. Мы определяем `device` в начале. torch.cuda.is_available() проверяет, есть ли у вас GPU с поддержкой CUDA.
# 2. Модели (.to(device)): Все нейросети (их веса и буферы) должны быть перемещены на GPU один раз при инициализации.
#    GPU выполняет матричные операции на порядки быстрее CPU.
# 3. Тензоры (.to(device)): Любые данные (тензоры), которые вы подаете в модель, должны быть на том же устройстве, что и модель.
#    Поэтому в цикле обучения мы перемещаем батчи из replay buffer на `device` перед вычислениями.
# Разница в коде минимальна, но в производительности - огромна. На CPU обучение будет идти нереально долго.

def main():
    # Загрузка конфигурации
    with open('base_params.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 1. Определение устройства (GPU или CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Инициализация среды, буфера и агента
    env = DummyTradingEnv(config)
    replay_buffer = ReplayBuffer(capacity=100000)
    agent = SACAgent(config, device)

    # Параметры обучения
    max_episodes = 500
    max_steps = 200
    batch_size = 256
    start_timesteps = 1000 # Количество "случайных" шагов для наполнения буфера

    total_steps = 0

    # --- Главный цикл обучения ---
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            total_steps += 1
            
            # На начальных этапах делаем случайные действия для исследования
            if total_steps < start_timesteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(torch.FloatTensor(state)).numpy()

            # Шаг в среде
            next_state, reward, done, _, _ = env.step(action)
            
            # Сохраняем переход в буфер
            replay_buffer.push(state, action, reward, next_state, float(done))

            state = next_state
            episode_reward += reward

            # Если буфер достаточно наполнен, начинаем обучение
            if len(replay_buffer) > batch_size and total_steps > start_timesteps:
                agent.update(replay_buffer, batch_size)

            if done:
                break
        
        print(f"Episode: {episode+1}, Total Steps: {total_steps}, Reward: {episode_reward:.2f}")

if __name__ == '__main__':
    main()