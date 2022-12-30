import gymnasium as gym
env = gym.make('CartPole-v1', render_mode='human')

observation, info = env.reset(seed=42)
env.render()
i = 0
while (i < 200):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f'{i} o: {observation}, r: {reward}, term: {terminated}, trunc: {truncated}, info: {info}')

    if terminated or truncated:
        i = i+1
        observation, info = env.reset()
env.close()