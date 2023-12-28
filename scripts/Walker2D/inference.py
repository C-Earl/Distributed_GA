import torch
from DDPG import TD3
import numpy as np
import gymnasium as gym
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

def evaluate_policy(env, policy, eval_episodes=10, video=None):
  avg_reward = 0.
  for _ in range(eval_episodes):
    obs = env.reset()[0]
    done = False
    while not done:
      if video is not None:
        video.capture_frame()
      action = policy.select_action(np.array(obs))
      obs, reward, term, done, _ = env.step(action)
      avg_reward += reward
  avg_reward /= eval_episodes
  # print ("---------------------------------------")
  # print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
  # print ("---------------------------------------")
  return avg_reward

if __name__ == "__main__":
  env_name = "Walker2d-v4"
  seed = 0

  file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
  print ("---------------------------------------")
  print ("Settings: %s" % (file_name))
  print ("---------------------------------------")

  eval_episodes = 10
  save_env_vid = True
  env = gym.make(env_name, render_mode='rgb_array', healthy_z_range=(0.8, 2.0))
  max_episode_steps = env._max_episode_steps
  if save_env_vid:
    video = VideoRecorder(env, path='model_run.mp4')
    # env = wrappers.monitoring.video_recorder(env, monitor_dir, force = True)
    env.reset()
  # env.seed(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0]
  max_action = float(env.action_space.high[0])
  policy = TD3(state_dim, action_dim, max_action)
  policy.load(file_name, './pytorch_models/')
  _ = evaluate_policy(policy, eval_episodes=eval_episodes, video=video)
  video.close()
